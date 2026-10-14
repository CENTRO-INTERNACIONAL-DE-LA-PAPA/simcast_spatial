"""Self-contained tests for prepare_inputs on a tiny synthetic grid."""

import netCDF4
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from prepare_inputs import (
    GridDef,
    build_td_from_predictions,
    extract_month_from_netcdf,
    rewrite_parquet_month,
)

GRID = GridDef(lat0=0.0, lon0=-80.0, step=0.1, nlat=4, nlon=5)


def _write_nc(path, var, dates, grid, values):
    with netCDF4.Dataset(path, "w") as nc:
        nc.createDimension("time", len(dates))
        nc.createDimension("latitude", grid.nlat)
        nc.createDimension("longitude", grid.nlon)
        t = nc.createVariable("time", "f8", ("time",))
        t.units = "days since 1970-01-01"
        t.calendar = "standard"
        t[:] = (pd.DatetimeIndex(dates) - pd.Timestamp("1970-01-01")).days
        la = nc.createVariable("latitude", "f8", ("latitude",))
        la[:] = grid.lat0 - np.arange(grid.nlat) * grid.step
        lo = nc.createVariable("longitude", "f8", ("longitude",))
        lo[:] = grid.lon0 + np.arange(grid.nlon) * grid.step
        v = nc.createVariable(var, "f4", ("time", "latitude", "longitude"), fill_value=np.nan)
        v[:] = values


def test_rowcol_and_gid():
    lon = np.array([-80.0, -79.6, -79.9])
    lat = np.array([0.0, -0.3, -0.1])
    row, col = GRID.rowcol(lon, lat)
    assert row.tolist() == [0, 3, 1]
    assert col.tolist() == [0, 4, 1]
    with pytest.raises(ValueError):
        GRID.rowcol(np.array([-79.95]), np.array([0.0]))  # off-node


def test_extract_month_from_yearly_netcdf(tmp_path):
    dates = pd.date_range("2001-01-01", "2001-03-31", freq="D")
    rng = np.random.default_rng(0)
    vals = rng.normal(size=(len(dates), GRID.nlat, GRID.nlon)).astype(np.float32)
    vals[:, 0, 0] = np.nan  # a masked cell
    nc = tmp_path / "tmax_daily_2001.nc"
    _write_nc(nc, "tmax", dates, GRID, vals)

    ids = np.array([0, 1, 2], dtype=np.int32)
    rows = np.array([0, 3, 1])
    cols = np.array([0, 4, 2])
    out = tmp_path / "tmax_daily_2001_02.parquet"
    extract_month_from_netcdf(
        nc, "tmax", 2001, 2, ids=ids, rows=rows, cols=cols, out_path=out, grid=GRID,
        row_group_ids=2,
    )
    df = pq.read_table(out).to_pandas()
    assert df.columns.tolist() == ["ID", "FECHA", "Value"]
    assert len(df) == 3 * 28
    assert df["ID"].is_monotonic_increasing  # ID-major
    feb = (dates.month == 2)
    for k, (r, c) in enumerate(zip(rows, cols)):
        sub = df[df.ID == k]
        assert (sub.FECHA.dt.day.to_numpy() == np.arange(1, 29)).all()
        expected = vals[feb, r, c]
        np.testing.assert_array_equal(sub.Value.to_numpy(), expected)
    assert df[df.ID == 0].Value.isna().all()
    # row-group statistics must allow pruning by ID
    md = pq.ParquetFile(out).metadata
    assert md.num_row_groups == 2
    assert md.row_group(0).column(0).statistics.max == 1

    with pytest.raises(ValueError):
        extract_month_from_netcdf(
            nc, "tmax", 2001, 7, ids=ids, rows=rows, cols=cols,
            out_path=tmp_path / "x.parquet", grid=GRID,
        )


def test_rewrite_parquet_month(tmp_path):
    dates = pd.date_range("1990-05-01", "1990-05-31")
    ids = np.array([5, 1, 3])
    # date-major, unsorted IDs, int64, timestamp[us]
    src = pd.DataFrame(
        {
            "ID": np.tile(ids, len(dates)).astype(np.int64),
            "FECHA": np.repeat(dates.to_numpy().astype("datetime64[us]"), len(ids)),
            "Value": np.arange(len(ids) * len(dates), dtype=np.float32),
            "source_file": "x.nc",
        }
    )
    p = tmp_path / "prec_daily_1990_05.parquet"
    src.to_parquet(p)
    out = tmp_path / "out.parquet"
    rewrite_parquet_month(p, out, expected_ids=np.array([1, 3, 5]))
    df = pd.read_parquet(out)
    assert df.ID.tolist() == sorted(df.ID.tolist())
    m = src.merge(df, on=["ID", "FECHA"], suffixes=("_a", "_b"))
    assert len(m) == len(src)
    np.testing.assert_array_equal(m.Value_a, m.Value_b)
    with pytest.raises(ValueError):
        rewrite_parquet_month(p, tmp_path / "y.parquet", expected_ids=np.array([1, 3]))


def test_build_td_from_predictions(tmp_path):
    n_buckets = 4
    mapping = pd.DataFrame(
        {"ID": np.array([0, 1, 2], dtype=np.int32), "gid": np.array([7, 9, 14])}
    )
    start = pd.Timestamp("2017-01-01")
    n_days = (pd.Timestamp("2018-12-31") - start).days + 1
    preds = tmp_path / "preds"
    rng = np.random.default_rng(1)
    truth = {}
    for b in range(n_buckets):
        gids = [g for g in range(GRID.size) if g % n_buckets == b]
        frames = []
        for g in gids:
            v = rng.normal(size=n_days).astype(np.float32)
            truth[g] = v
            frames.append(
                pd.DataFrame(
                    {"ID": np.int32(g), "day": np.arange(n_days, dtype=np.int16),
                     "pred": v, "q50": v}
                )
            )
        d = preds / f"id_bucket={b:04d}"
        d.mkdir(parents=True)
        pd.concat(frames).to_parquet(d / "pred.parquet", index=False)

    out = tmp_path / "td"
    files = build_td_from_predictions(
        mapping, preds_dir=preds, out_dir=out, start="2017-01-01",
        years=(2017, 2018), n_buckets=n_buckets, grid=GRID, workers=2,
    )
    assert len(files) == 24
    df = pd.read_parquet(out / "td_daily_2018_02.parquet")
    assert len(df) == 3 * 28
    for pid, g in zip(mapping.ID, mapping.gid):
        sub = df[df.ID == pid]
        d0 = (pd.Timestamp("2018-02-01") - start).days
        np.testing.assert_array_equal(sub.Value.to_numpy(), truth[g][d0 : d0 + 28])
        assert sub.FECHA.iloc[0] == pd.Timestamp("2018-02-01")

    # a mapping gid outside the panel must fail loudly
    bad = pd.DataFrame({"ID": [0], "gid": [GRID.size - 1]})
    (preds / "id_bucket=0003" / "pred.parquet").unlink()
    pd.DataFrame({"ID": [3], "day": [0], "pred": [1.0]}).to_parquet(
        preds / "id_bucket=0003" / "pred.parquet"
    )
    with pytest.raises(RuntimeError):
        build_td_from_predictions(
            bad, preds_dir=preds, out_dir=tmp_path / "td2", years=(2017, 2018),
            n_buckets=n_buckets, grid=GRID, workers=1,
        )


def test_bilinear_weights_and_values():
    from prepare_inputs import bilinear_at_points, bilinear_weights

    lon = np.array([-80.05, -79.95, -79.85])  # centres, ascending
    lat = np.array([0.05, -0.05, -0.15])  # descending
    field = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.float32)
    # f(lon, lat) is linear: 1 + 10*(lon+80.05) + 30*(0.05-lat)
    px = np.array([-80.05, -80.0, -79.9, -79.85, -79.93])
    py = np.array([0.05, 0.0, -0.1, -0.15, -0.02])
    i0, j0, w = bilinear_weights(lon, lat, px, py)
    np.testing.assert_allclose(w.sum(1), 1.0)
    got = bilinear_at_points(field, i0, j0, w)[0]
    expected = 1 + 10 * (px + 80.05) + 30 * (0.05 - py)
    np.testing.assert_allclose(got, expected, rtol=1e-5)


def test_bilinear_nan_policies():
    from prepare_inputs import bilinear_at_points, bilinear_weights

    lon = np.array([0.0, 1.0])
    lat = np.array([1.0, 0.0])
    field = np.array([[[np.nan, 2.0], [2.0, 2.0]]], dtype=np.float32)
    i0, j0, w = bilinear_weights(lon, lat, np.array([0.5, 1.0]), np.array([0.5, 0.0]))
    ren = bilinear_at_points(field, i0, j0, w, nan_policy="renormalize")[0]
    pro = bilinear_at_points(field, i0, j0, w, nan_policy="propagate")[0]
    assert ren[0] == pytest.approx(2.0)  # NaN corner dropped, weights rescaled
    assert np.isnan(pro[0])
    # point on a valid corner: the NaN corner has zero weight -> not NaN in either
    assert ren[1] == pytest.approx(2.0) and pro[1] == pytest.approx(2.0)


def test_extract_prec_bilinear_index_time(tmp_path):
    from prepare_inputs import extract_prec_bilinear

    n_t = 59  # 2001-01-01 .. 2001-02-28
    lon = np.array([-80.05, -79.95, -79.85])
    lat = np.array([0.05, -0.05, -0.15])
    nc_path = tmp_path / "p.nc"
    with netCDF4.Dataset(nc_path, "w") as nc:
        nc.createDimension("Z1", n_t)
        nc.createDimension("latitude", 3)
        nc.createDimension("longitude", 3)
        z = nc.createVariable("Z1", "i4", ("Z1",))
        z.units = "unknown"
        z[:] = np.arange(1, n_t + 1)
        nc.createVariable("latitude", "f8", ("latitude",))[:] = lat
        nc.createVariable("longitude", "f8", ("longitude",))[:] = lon
        v = nc.createVariable("precipitation", "f4", ("Z1", "latitude", "longitude"),
                              fill_value=np.float32(-1.175494e38))
        data = np.arange(n_t, dtype=np.float32)[:, None, None] * np.ones((1, 3, 3), np.float32)
        v[:] = data
    mapping = pd.DataFrame({"ID": [1, 0], "lon": [-79.95, -80.0], "lat": [-0.05, 0.0]})
    files = extract_prec_bilinear(nc_path, mapping, tmp_path / "out", start="2001-01-01",
                                  years=(2001, 2001), workers=1)
    assert len(files) == 2  # only Jan and Feb have data
    feb = pd.read_parquet(tmp_path / "out" / "prec_daily_2001_02.parquet")
    assert feb.ID.tolist()[:2] == [0, 0]
    assert feb.FECHA.iloc[0] == pd.Timestamp("2001-02-01")
    np.testing.assert_allclose(feb[feb.ID == 1].Value.to_numpy(), np.arange(31, 59))
