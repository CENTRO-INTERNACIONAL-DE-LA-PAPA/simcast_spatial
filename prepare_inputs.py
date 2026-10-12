"""
Build the canonical SIMCAST input layout from the 2026 data sources.

Target layout (what ``load_variables.py`` reads)::

    <output_base>/
        grid_mapping.parquet            potato ID <-> PISCO grid cell (gid, row, col, lon, lat, Type)
        provenance.json                 where every variable came from
        prec/Outputs/prec_daily_YYYY_MM.parquet
        tmax/Outputs/tmax_daily_YYYY_MM.parquet
        tmin/Outputs/tmin_daily_YYYY_MM.parquet
        td/Outputs/td_daily_YYYY_MM.parquet

Every monthly parquet has the columns ``ID`` (int32, potato-grid index),
``FECHA`` (timestamp, midnight) and ``Value`` (float32), is sorted by
``(ID, FECHA)`` and is written with ~5 000 IDs per row group so that the
``ID`` range filters used by ``load_variables.load_variable_data`` prune row
groups through parquet statistics.

Sources
-------
* ``tmax`` / ``tmin``: PISCOt v1.2 yearly NetCDFs (``{var}_daily_YYYY.nc``, 1981-2020).
* ``td`` 1981-2016: PISCOeo_pm monthly NetCDFs (``td_daily_YYYY_MM.nc``).
* ``td`` 2017-2020: TimesFM predictions (``id_bucket=NNNN/pred.parquet``,
  columns ``ID`` (PISCO gid), ``day`` (0 = 2017-01-01), ``pred``).
* ``prec`` (preferred): a single daily 0.1 deg NetCDF such as PISCOp v3.0
  (``PISCOp_d.nc``, 1981-2025), bilinearly interpolated straight to the
  potato points (see ``extract_prec_bilinear``).  This is the
  "resample to 0.01 deg + pick the node" step of ``resampling.py`` +
  ``masking.py`` done in one pass.
* ``prec`` (legacy): the monthly parquets produced earlier by ``masking.py``
  from PISCOp v2 (1981-2019), re-written in the new sort order / dtypes.

All NetCDF extraction is done by *exact grid-cell indexing*: every potato
point sits on a node of the 0.01 deg PISCO grid, so the value is
``var[t, row, col]`` with no interpolation.  This reproduces bit-for-bit the
parquets generated earlier with ``masking.py`` (nearest neighbour).
"""

from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Grid definition
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class GridDef:
    """Regular lat/lon grid; ``gid = row * nlon + col`` with row 0 = northernmost."""

    lat0: float = 1.22
    lon0: float = -81.40
    step: float = 0.01
    nlat: int = 1982
    nlon: int = 1422

    @property
    def size(self) -> int:
        return self.nlat * self.nlon

    def rowcol(self, lon: np.ndarray, lat: np.ndarray, tol: float = 1e-5):
        col_f = (np.asarray(lon, dtype=np.float64) - self.lon0) / self.step
        row_f = (self.lat0 - np.asarray(lat, dtype=np.float64)) / self.step
        col = np.rint(col_f).astype(np.int64)
        row = np.rint(row_f).astype(np.int64)
        off = max(np.abs(col_f - col).max(), np.abs(row_f - row).max())
        if off > tol:
            raise ValueError(
                f"Points are not on the grid nodes (max offset {off:.2e} cells)"
            )
        if (col < 0).any() or (col >= self.nlon).any() or (row < 0).any() or (
            row >= self.nlat
        ).any():
            raise ValueError("Points fall outside the grid domain")
        return row, col


PISCO_GRID = GridDef()

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class InputSources:
    potato_grid_file: Path
    output_base: Path

    tmax_nc_dir: Path | None = None  # yearly tmax_daily_YYYY.nc
    tmin_nc_dir: Path | None = None  # yearly tmin_daily_YYYY.nc
    td_nc_dir: Path | None = None  # monthly td_daily_YYYY_MM.nc
    prec_parquet_dir: Path | None = None  # prec_daily_YYYY_MM.parquet (potato IDs)
    prec_nc_file: Path | None = None  # one daily NetCDF (e.g. PISCOp v3.0 PISCOp_d.nc)
    prec_nc_var: str = "precipitation"
    prec_nc_start: str = "1981-01-01"  # date of the first time step (time axis is an index)
    prec_years: tuple[int, int] | None = None  # defaults to `years`
    prec_nan_policy: str = "renormalize"  # or "propagate" (any NaN neighbour -> NaN)

    # Variables to take, as symlinks, from an already built input tree
    # (e.g. reuse tmax/tmin/td of simcast_inputs_v2 when only prec changes).
    link_from: Path | None = None
    link_variables: tuple[str, ...] = ()

    td_pred_dir: Path | None = None  # id_bucket=NNNN/pred.parquet
    td_pred_start: str = "2017-01-01"
    td_pred_n_buckets: int = 8192
    td_pred_column: str = "pred"
    td_pred_years: tuple[int, int] = (2017, 2020)

    years: tuple[int, int] = (1981, 2020)
    td_nc_years: tuple[int, int] = (1981, 2016)

    grid: GridDef = field(default_factory=GridDef)
    grid_index_file: Path | None = None  # optional cross-check (ID, latitude, longitude)

    row_group_ids: int = 5000
    compression: str = "zstd"


PARQUET_SCHEMA = pa.schema(
    [
        ("ID", pa.int32()),
        ("FECHA", pa.timestamp("ns")),
        ("Value", pa.float32()),
    ]
)

# ----------------------------------------------------------------------------
# Grid mapping
# ----------------------------------------------------------------------------


def build_grid_mapping(
    potato_grid_file: str | os.PathLike[str],
    *,
    grid: GridDef = PISCO_GRID,
    grid_index_file: str | os.PathLike[str] | None = None,
) -> pd.DataFrame:
    """
    Map every potato point to its PISCO grid cell.

    The potato ``ID`` is the row index of the shapefile (this is what
    ``masking.py`` and ``load_variables.py`` use), unless an ``ID`` column exists.
    """
    import geopandas as gpd

    g = gpd.read_file(os.fspath(potato_grid_file))
    geom = g.geometry.apply(lambda s: s.centroid if s.geom_type != "Point" else s)
    lon = geom.x.to_numpy()
    lat = geom.y.to_numpy()
    row, col = grid.rowcol(lon, lat)
    gid = row * grid.nlon + col
    if len(np.unique(gid)) != len(gid):
        raise ValueError("Several potato points map to the same grid cell")

    ids = g["ID"].to_numpy() if "ID" in g.columns else g.index.to_numpy()
    out = pd.DataFrame(
        {
            "ID": ids.astype(np.int32),
            "gid": gid.astype(np.int64),
            "row": row.astype(np.int32),
            "col": col.astype(np.int32),
            "lon": lon,
            "lat": lat,
        }
    )
    if "Type" in g.columns:
        out["Type"] = g["Type"].to_numpy()

    if grid_index_file is not None:
        gi = pd.read_parquet(os.fspath(grid_index_file)).set_index("ID")
        lat_col = "latitude" if "latitude" in gi.columns else "lat"
        lon_col = "longitude" if "longitude" in gi.columns else "lon"
        ref = gi.loc[out["gid"].to_numpy()]
        d = max(
            np.abs(ref[lat_col].to_numpy() - lat).max(),
            np.abs(ref[lon_col].to_numpy() - lon).max(),
        )
        if d > 1e-6:
            raise ValueError(f"grid_index cross-check failed (max offset {d:.2e} deg)")
    return out.sort_values("ID").reset_index(drop=True)


# ----------------------------------------------------------------------------
# Parquet writing helpers
# ----------------------------------------------------------------------------


def _write_month(
    out_path: Path,
    ids: np.ndarray,
    fechas: np.ndarray,
    values: np.ndarray,
    *,
    row_group_ids: int,
    compression: str,
) -> None:
    """Write an ID-major monthly parquet atomically (tmp + rename)."""
    n_days = len(np.unique(fechas))
    table = pa.Table.from_arrays(
        [
            pa.array(np.asarray(ids, dtype=np.int32)),
            pa.array(np.asarray(fechas, dtype="datetime64[ns]")),
            pa.array(np.asarray(values, dtype=np.float32)),
        ],
        schema=PARQUET_SCHEMA,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    pq.write_table(
        table,
        tmp,
        row_group_size=max(1, row_group_ids) * max(1, n_days),
        compression=compression,
    )
    os.replace(tmp, out_path)


def _month_dates(year: int, month: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(year=year, month=month, day=1)
    return pd.date_range(start, start + pd.offsets.MonthEnd(0), freq="D")


# ----------------------------------------------------------------------------
# NetCDF extraction (exact cell indexing)
# ----------------------------------------------------------------------------


def _nc_times(nc: netCDF4.Dataset, name: str = "time") -> pd.DatetimeIndex:
    tv = nc.variables[name]
    cal = getattr(tv, "calendar", "standard")
    t = netCDF4.num2date(
        tv[:], tv.units, calendar=cal, only_use_cftime_datetimes=False
    )
    return pd.DatetimeIndex(pd.to_datetime(list(t))).normalize()


def _check_nc_grid(nc: netCDF4.Dataset, grid: GridDef) -> tuple[str, str]:
    lat_name = "latitude" if "latitude" in nc.variables else "lat"
    lon_name = "longitude" if "longitude" in nc.variables else "lon"
    lat = nc.variables[lat_name]
    lon = nc.variables[lon_name]
    if lat.shape[0] != grid.nlat or lon.shape[0] != grid.nlon:
        raise ValueError(
            f"NetCDF grid {lat.shape[0]}x{lon.shape[0]} != expected {grid.nlat}x{grid.nlon}"
        )
    if abs(float(lat[0]) - grid.lat0) > 1e-6 or abs(float(lon[0]) - grid.lon0) > 1e-6:
        raise ValueError(
            f"NetCDF origin ({float(lat[0])}, {float(lon[0])}) != expected ({grid.lat0}, {grid.lon0})"
        )
    return lat_name, lon_name


def extract_month_from_netcdf(
    nc_path: str | os.PathLike[str],
    var_name: str,
    year: int,
    month: int,
    *,
    ids: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    out_path: str | os.PathLike[str],
    grid: GridDef = PISCO_GRID,
    row_group_ids: int = 5000,
    compression: str = "zstd",
) -> str:
    """Extract one calendar month for all points from a (yearly or monthly) NetCDF."""
    out_path = Path(out_path)
    with netCDF4.Dataset(os.fspath(nc_path)) as nc:
        var = nc.variables[var_name]
        lat_name, lon_name = _check_nc_grid(nc, grid)
        dims = var.dimensions
        if dims[-2] != lat_name or dims[-1] != lon_name:
            raise ValueError(f"Unexpected dimension order {dims} in {nc_path}")
        times = _nc_times(nc, dims[0])
        idx = np.flatnonzero((times.year == year) & (times.month == month))
        if idx.size == 0:
            raise ValueError(f"{nc_path} has no data for {year}-{month:02d}")
        if idx[-1] - idx[0] + 1 != idx.size:
            raise ValueError(f"{nc_path}: month {year}-{month:02d} is not contiguous")
        block = var[idx[0] : idx[-1] + 1, :, :]
        block = np.ma.filled(np.ma.asarray(block, dtype=np.float32), np.nan)
        month_times = times[idx]

    vals = block[:, rows, cols]  # (n_days, n_pts)
    n_days, n_pts = vals.shape
    _write_month(
        out_path,
        np.repeat(ids, n_days),
        np.tile(month_times.to_numpy(), n_pts),
        np.ascontiguousarray(vals.T).ravel(),
        row_group_ids=row_group_ids,
        compression=compression,
    )
    return os.fspath(out_path)


# ----------------------------------------------------------------------------
# Prec: re-write existing potato-ID parquets in the new layout
# ----------------------------------------------------------------------------


def rewrite_parquet_month(
    src: str | os.PathLike[str],
    out_path: str | os.PathLike[str],
    *,
    expected_ids: np.ndarray | None = None,
    row_group_ids: int = 5000,
    compression: str = "zstd",
) -> str:
    t = pq.read_table(os.fspath(src), columns=["ID", "FECHA", "Value"])
    ids = t.column("ID").to_numpy()
    fechas = t.column("FECHA").to_numpy().astype("datetime64[ns]")
    fechas = fechas.astype("datetime64[D]").astype("datetime64[ns]")  # normalize
    vals = t.column("Value").to_numpy(zero_copy_only=False)
    if expected_ids is not None:
        extra = np.setdiff1d(np.unique(ids), expected_ids)
        if extra.size:
            raise ValueError(f"{src}: {extra.size} unexpected IDs (e.g. {extra[:3]})")
    order = np.lexsort((fechas, ids))
    _write_month(
        Path(out_path),
        ids[order],
        fechas[order],
        vals[order],
        row_group_ids=row_group_ids,
        compression=compression,
    )
    return os.fspath(out_path)


# ----------------------------------------------------------------------------
# td from TimesFM predictions (bucketed parquet)
# ----------------------------------------------------------------------------


def _read_pred_bucket(path: str, wanted_gids: np.ndarray, column: str):
    t = pq.read_table(path, columns=["ID", "day", column])
    gid = t.column("ID").to_numpy().astype(np.int64)
    mask = np.isin(gid, wanted_gids)
    return (
        gid[mask],
        t.column("day").to_numpy()[mask].astype(np.int32),
        t.column(column).to_numpy(zero_copy_only=False)[mask].astype(np.float32),
    )


def build_td_from_predictions(
    mapping: pd.DataFrame,
    *,
    preds_dir: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    start: str = "2017-01-01",
    years: tuple[int, int] = (2017, 2020),
    n_buckets: int = 8192,
    column: str = "pred",
    variable: str = "td",
    grid: GridDef = PISCO_GRID,
    workers: int = 8,
    overwrite: bool = False,
    row_group_ids: int = 5000,
    compression: str = "zstd",
) -> list[str]:
    """
    Convert ``id_bucket=NNNN/pred.parquet`` (ID = PISCO gid, day index) into
    monthly potato-ID parquets.  Bucket assignment is ``gid % n_buckets``.
    """
    preds_dir = Path(os.fspath(preds_dir))
    out_dir = Path(os.fspath(out_dir))
    start_ts = pd.Timestamp(start)
    y0, y1 = years
    months = [(y, m) for y in range(y0, y1 + 1) for m in range(1, 13)]
    outputs = {
        (y, m): out_dir / f"{variable}_daily_{y}_{m:02d}.parquet" for (y, m) in months
    }
    if not overwrite and all(p.exists() for p in outputs.values()):
        return [os.fspath(p) for p in outputs.values()]

    gids = mapping["gid"].to_numpy().astype(np.int64)
    lut = np.full(grid.size, -1, dtype=np.int32)
    lut[gids] = mapping["ID"].to_numpy().astype(np.int32)

    by_bucket = {}
    for b, sub in pd.Series(gids).groupby(gids % n_buckets):
        by_bucket[int(b)] = sub.to_numpy()

    jobs = [
        (os.fspath(preds_dir / f"id_bucket={b:04d}" / "pred.parquet"), want)
        for b, want in sorted(by_bucket.items())
    ]
    missing = [p for p, _ in jobs if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} bucket files missing, e.g. {missing[:3]}")

    g_parts, d_parts, v_parts = [], [], []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_read_pred_bucket, p, w, column) for p, w in jobs]
        for f in tqdm(futs, desc=f"{variable} predictions", unit="bucket"):
            g, d, v = f.result()
            g_parts.append(g)
            d_parts.append(d)
            v_parts.append(v)
    gid_all = np.concatenate(g_parts)
    day_all = np.concatenate(d_parts)
    val_all = np.concatenate(v_parts)
    del g_parts, d_parts, v_parts

    pid_all = lut[gid_all]
    if (pid_all < 0).any():
        raise RuntimeError("prediction rows for gids outside the potato grid")

    first_day = (pd.Timestamp(year=y0, month=1, day=1) - start_ts).days
    last_day = (pd.Timestamp(year=y1, month=12, day=31) - start_ts).days
    counts = np.bincount(pid_all, minlength=len(mapping))
    expected = last_day - first_day + 1
    if (counts < expected).any():
        bad = int((counts < expected).sum())
        raise RuntimeError(
            f"{bad} potato IDs have fewer than {expected} predicted days "
            f"(min {counts.min()})"
        )

    written = []
    for y, m in tqdm(months, desc=f"{variable} monthly files", unit="month"):
        out_path = outputs[(y, m)]
        if out_path.exists() and not overwrite:
            written.append(os.fspath(out_path))
            continue
        dates = _month_dates(y, m)
        d0 = (dates[0] - start_ts).days
        d1 = (dates[-1] - start_ts).days
        sel = np.flatnonzero((day_all >= d0) & (day_all <= d1))
        pid = pid_all[sel]
        day = day_all[sel]
        order = np.lexsort((day, pid))
        fechas = (start_ts.to_datetime64() + day[order].astype("timedelta64[D]")).astype(
            "datetime64[ns]"
        )
        _write_month(
            out_path,
            pid[order],
            fechas,
            val_all[sel][order],
            row_group_ids=row_group_ids,
            compression=compression,
        )
        written.append(os.fspath(out_path))
    return written


# ----------------------------------------------------------------------------
# prec from a coarse daily NetCDF: bilinear interpolation at the points
# ----------------------------------------------------------------------------


def bilinear_weights(
    src_lon: np.ndarray, src_lat: np.ndarray, px: np.ndarray, py: np.ndarray
):
    """
    Indices and weights of the 2x2 source cells around every point.

    ``src_lon`` / ``src_lat`` are the (regular) cell-centre coordinates of the
    source grid; latitude may be descending.  Points outside the outer ring of
    centres are clamped to the edge (same as GDAL).
    Returns ``(i0, j0, w)`` with ``w`` shaped (n_points, 4) for the corners
    (i0,j0), (i0,j0+1), (i0+1,j0), (i0+1,j0+1).
    """
    src_lon = np.asarray(src_lon, dtype=np.float64)
    src_lat = np.asarray(src_lat, dtype=np.float64)
    dx = (src_lon[-1] - src_lon[0]) / (len(src_lon) - 1)
    dy = (src_lat[-1] - src_lat[0]) / (len(src_lat) - 1)
    fx = (np.asarray(px, dtype=np.float64) - src_lon[0]) / dx
    fy = (np.asarray(py, dtype=np.float64) - src_lat[0]) / dy
    # snap values within 1e-6 cells of an integer (points on cell centres)
    fx = np.where(np.abs(fx - np.rint(fx)) < 1e-6, np.rint(fx), fx)
    fy = np.where(np.abs(fy - np.rint(fy)) < 1e-6, np.rint(fy), fy)
    j0 = np.clip(np.floor(fx).astype(np.int64), 0, len(src_lon) - 2)
    i0 = np.clip(np.floor(fy).astype(np.int64), 0, len(src_lat) - 2)
    wx = np.clip(fx - j0, 0.0, 1.0)
    wy = np.clip(fy - i0, 0.0, 1.0)
    w = np.stack(
        [(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx], axis=1
    )
    return i0, j0, w


def bilinear_at_points(
    field: np.ndarray,
    i0: np.ndarray,
    j0: np.ndarray,
    w: np.ndarray,
    *,
    nan_policy: str = "renormalize",
) -> np.ndarray:
    """
    Interpolate ``field`` (time, lat, lon) at the points -> (time, n_points).

    ``nan_policy="renormalize"``: NaN corners (sea, outside the product) are
    dropped and the remaining weights rescaled; NaN only when every corner
    with a non-zero weight is NaN.  ``"propagate"``: NaN when any corner with
    a non-zero weight is NaN.
    """
    corners = (
        field[:, i0, j0],
        field[:, i0, j0 + 1],
        field[:, i0 + 1, j0],
        field[:, i0 + 1, j0 + 1],
    )
    num = np.zeros(corners[0].shape, dtype=np.float64)
    den = np.zeros(corners[0].shape, dtype=np.float64)
    any_nan = np.zeros(corners[0].shape, dtype=bool)
    for k, c in enumerate(corners):
        wk = w[:, k][None, :]
        bad = np.isnan(c)
        use = (~bad) & (wk > 0)
        num += np.where(use, wk * np.where(bad, 0.0, c), 0.0)
        den += np.where(use, wk, 0.0)
        any_nan |= bad & (wk > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den <= 0] = np.nan
    if nan_policy == "propagate":
        out[any_nan] = np.nan
    elif nan_policy != "renormalize":
        raise ValueError("nan_policy must be 'renormalize' or 'propagate'")
    return out.astype(np.float32)


def extract_prec_bilinear(
    nc_file: str | os.PathLike[str],
    mapping: pd.DataFrame,
    out_dir: str | os.PathLike[str],
    *,
    var_name: str = "precipitation",
    start: str = "1981-01-01",
    years: tuple[int, int] = (1981, 2025),
    variable: str = "prec",
    nan_policy: str = "renormalize",
    overwrite: bool = False,
    workers: int = 8,
    row_group_ids: int = 5000,
    compression: str = "zstd",
) -> list[str]:
    """
    Bilinearly interpolate a daily (time, lat, lon) NetCDF to the potato
    points and write monthly ``<variable>_daily_YYYY_MM.parquet``.

    The time axis may be a plain index (PISCOp v3.0 ``Z1`` = 1..N); dates
    are ``start + k`` days.  If the time variable has CF units they are used
    instead.  The whole cube is read once (PISCOp v3.0: 2.4 GB).

    Equivalent to ``resampling.py`` (GDAL bilinear to the 0.01 deg grid) +
    ``masking.py`` (pick the node): on PISCOp v2 it reproduces the old prec
    parquets to < 1e-5 mm wherever all four neighbours are land.
    """
    out_dir = Path(os.fspath(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    y0, y1 = years
    months = [(y, m) for y in range(y0, y1 + 1) for m in range(1, 13)]
    outputs = {(y, m): out_dir / f"{variable}_daily_{y}_{m:02d}.parquet" for y, m in months}
    todo = [k for k, p in outputs.items() if overwrite or not p.exists()]
    if not todo:
        return [os.fspath(p) for p in outputs.values()]

    with netCDF4.Dataset(os.fspath(nc_file)) as nc:
        var = nc.variables[var_name]
        tdim, ydim, xdim = var.dimensions
        lat_name = ydim if ydim in nc.variables else ("latitude" if "latitude" in nc.variables else "lat")
        lon_name = xdim if xdim in nc.variables else ("longitude" if "longitude" in nc.variables else "lon")
        src_lat = nc.variables[lat_name][:].astype(np.float64)
        src_lon = nc.variables[lon_name][:].astype(np.float64)
        n_t = var.shape[0]
        tv = nc.variables.get(tdim)
        units = getattr(tv, "units", "") if tv is not None else ""
        if " since " in units:
            dates = _nc_times(nc, tdim)
        else:
            dates = pd.date_range(pd.Timestamp(start), periods=n_t, freq="D")
        print(f"[prec] reading {nc_file} ({n_t} days, {dates[0].date()} .. {dates[-1].date()})")
        cube = np.ma.filled(np.ma.asarray(var[:, :, :], dtype=np.float32), np.nan)
    cube[~np.isfinite(cube)] = np.nan
    cube[np.abs(cube) > 1e30] = np.nan  # e.g. _FillValue -1.17e38 if not masked
    neg = int((cube < 0).sum())
    if neg:
        print(f"[prec] WARNING {neg} negative values set to 0")
        cube[cube < 0] = 0.0

    # check the points are inside the source domain
    px = mapping["lon"].to_numpy()
    py = mapping["lat"].to_numpy()
    if px.min() < src_lon.min() or px.max() > src_lon.max() or py.min() < min(
        src_lat.min(), src_lat.max()
    ) or py.max() > max(src_lat.min(), src_lat.max()):
        raise ValueError("Some points fall outside the precipitation grid")

    i0, j0, w = bilinear_weights(src_lon, src_lat, px, py)
    ids = mapping["ID"].to_numpy().astype(np.int32)
    order = np.argsort(ids, kind="stable")

    def _one(ym):
        y, m = ym
        sel = np.flatnonzero((dates.year == y) & (dates.month == m))
        if sel.size == 0:
            return None
        vals = bilinear_at_points(cube[sel], i0, j0, w, nan_policy=nan_policy)[:, order]
        n_days, n_pts = vals.shape
        _write_month(
            outputs[ym],
            np.repeat(ids[order], n_days),
            np.tile(dates[sel].to_numpy(), n_pts),
            np.ascontiguousarray(vals.T).ravel(),
            row_group_ids=row_group_ids,
            compression=compression,
        )
        return os.fspath(outputs[ym])

    written = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for r in tqdm(ex.map(_one, todo), total=len(todo), desc=f"{variable} bilinear"):
            if r:
                written.append(r)
    missing = [f"{y}-{m:02d}" for (y, m) in todo if not outputs[(y, m)].exists()]
    if missing:
        print(f"[prec] no source data for {len(missing)} months (first {missing[0]})")
    return written


def link_variables(
    base: str | os.PathLike[str],
    other_base: str | os.PathLike[str],
    variables,
) -> None:
    """Make ``<base>/<var>`` a symlink to ``<other_base>/<var>`` (reuse inputs)."""
    base = Path(os.fspath(base))
    other_base = Path(os.fspath(other_base))
    for v in variables:
        target = other_base / v
        link = base / v
        if not target.is_dir():
            raise FileNotFoundError(target)
        if link.is_symlink():
            if Path(os.readlink(link)) == target:
                continue
            link.unlink()
        elif link.exists():
            raise FileExistsError(f"{link} exists and is not a symlink")
        link.symlink_to(target, target_is_directory=True)
        print(f"[prepare] {link} -> {target}")


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------


def _month_range(years: tuple[int, int]):
    y0, y1 = years
    return [(y, m) for y in range(y0, y1 + 1) for m in range(1, 13)]


def check_layout(
    base: str | os.PathLike[str],
    variables=("prec", "tmax", "tmin", "td"),
    years: tuple[int, int] = (1981, 2020),
) -> dict[str, dict]:
    """Report which monthly files exist per variable (and the gaps)."""
    base = Path(os.fspath(base))
    report = {}
    for v in variables:
        present, missing = [], []
        for y, m in _month_range(years):
            p = base / v / "Outputs" / f"{v}_daily_{y}_{m:02d}.parquet"
            (present if p.exists() else missing).append(f"{y}-{m:02d}")
        report[v] = {
            "n_present": len(present),
            "first": present[0] if present else None,
            "last": present[-1] if present else None,
            "missing": missing,
        }
    return report


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).parent,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def prepare_all(
    src: InputSources,
    *,
    workers: int = 8,
    overwrite: bool = False,
    variables=("prec", "tmax", "tmin", "td"),
) -> dict:
    """
    Build every monthly input parquet that does not exist yet.

    Returns the layout report (see ``check_layout``).  Each worker reads one
    calendar month (~350 MB of NetCDF) so RAM use is ~1 GB per worker.
    """
    base = Path(os.fspath(src.output_base))
    base.mkdir(parents=True, exist_ok=True)

    if src.link_from is not None and src.link_variables:
        link_variables(base, src.link_from, src.link_variables)
        variables = tuple(v for v in variables if v not in src.link_variables)

    mapping_path = base / "grid_mapping.parquet"
    mapping = build_grid_mapping(
        src.potato_grid_file, grid=src.grid, grid_index_file=src.grid_index_file
    )
    mapping.to_parquet(mapping_path, index=False)
    ids = mapping["ID"].to_numpy().astype(np.int32)
    rows = mapping["row"].to_numpy()
    cols = mapping["col"].to_numpy()
    print(f"[prepare] {len(mapping)} potato points mapped to grid cells -> {mapping_path}")

    nc_kwargs = dict(
        ids=ids,
        rows=rows,
        cols=cols,
        grid=src.grid,
        row_group_ids=src.row_group_ids,
        compression=src.compression,
    )
    jobs: list[tuple] = []  # (fn, args, kwargs, label)

    def _out(var: str, y: int, m: int) -> Path:
        return base / var / "Outputs" / f"{var}_daily_{y}_{m:02d}.parquet"

    # tmax / tmin: yearly NetCDFs
    for var, nc_dir in (("tmax", src.tmax_nc_dir), ("tmin", src.tmin_nc_dir)):
        if var not in variables or nc_dir is None:
            continue
        for y, m in _month_range(src.years):
            out = _out(var, y, m)
            if out.exists() and not overwrite:
                continue
            nc = Path(nc_dir) / f"{var}_daily_{y}.nc"
            if not nc.exists():
                print(f"[prepare] WARNING missing {nc}; skipping {var} {y}")
                continue
            jobs.append(
                (
                    extract_month_from_netcdf,
                    (os.fspath(nc), var, y, m),
                    {**nc_kwargs, "out_path": out},
                    f"{var} {y}-{m:02d}",
                )
            )

    # td <= 2016: monthly NetCDFs
    if "td" in variables and src.td_nc_dir is not None:
        for y, m in _month_range(src.td_nc_years):
            out = _out("td", y, m)
            if out.exists() and not overwrite:
                continue
            nc = Path(src.td_nc_dir) / f"td_daily_{y}_{m:02d}.nc"
            if not nc.exists():
                print(f"[prepare] WARNING missing {nc}; skipping td {y}-{m:02d}")
                continue
            jobs.append(
                (
                    extract_month_from_netcdf,
                    (os.fspath(nc), "td", y, m),
                    {**nc_kwargs, "out_path": out},
                    f"td {y}-{m:02d}",
                )
            )

    # prec: re-write existing parquets (legacy; ignored when prec_nc_file is set)
    if "prec" in variables and src.prec_parquet_dir is not None and src.prec_nc_file is None:
        for y, m in _month_range(src.years):
            out = _out("prec", y, m)
            if out.exists() and not overwrite:
                continue
            p = Path(src.prec_parquet_dir) / f"prec_daily_{y}_{m:02d}.parquet"
            if not p.exists():
                continue  # prec ends in 2019-12; gaps are reported by check_layout
            jobs.append(
                (
                    rewrite_parquet_month,
                    (os.fspath(p), out),
                    {
                        "expected_ids": ids,
                        "row_group_ids": src.row_group_ids,
                        "compression": src.compression,
                    },
                    f"prec {y}-{m:02d}",
                )
            )

    print(f"[prepare] {len(jobs)} monthly extraction jobs")
    failures = []
    if jobs:
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(fn, *args, **kwargs): label for fn, args, kwargs, label in jobs
            }
            for f in tqdm(cf.as_completed(futs), total=len(futs), desc="extract"):
                label = futs[f]
                try:
                    f.result()
                except Exception as e:  # keep going, report at the end
                    failures.append((label, repr(e)))
                    print(f"[prepare] FAILED {label}: {e!r}")

    # prec from a daily NetCDF (PISCOp v3.0): bilinear at the points
    if "prec" in variables and src.prec_nc_file is not None:
        extract_prec_bilinear(
            src.prec_nc_file,
            mapping,
            base / "prec" / "Outputs",
            var_name=src.prec_nc_var,
            start=src.prec_nc_start,
            years=src.prec_years or src.years,
            nan_policy=src.prec_nan_policy,
            overwrite=overwrite,
            workers=workers,
            row_group_ids=src.row_group_ids,
            compression=src.compression,
        )

    # td 2017-2020 from predictions
    if "td" in variables and src.td_pred_dir is not None:
        build_td_from_predictions(
            mapping,
            preds_dir=src.td_pred_dir,
            out_dir=base / "td" / "Outputs",
            start=src.td_pred_start,
            years=src.td_pred_years,
            n_buckets=src.td_pred_n_buckets,
            column=src.td_pred_column,
            grid=src.grid,
            workers=workers,
            overwrite=overwrite,
            row_group_ids=src.row_group_ids,
            compression=src.compression,
        )

    all_vars = tuple(dict.fromkeys((*variables, *src.link_variables)))
    report = check_layout(base, variables=all_vars, years=src.years)
    prov = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "sources": {k: (os.fspath(v) if isinstance(v, Path) else v) for k, v in asdict(src).items()},
        "n_points": int(len(mapping)),
        "layout": report,
        "failures": failures,
        "notes": (
            "tmax/tmin: PISCOt v1.2 (Huerta et al. 2023), exact cell indexing. "
            "td 1981-2016: PISCOeo_pm dew point (Huerta et al. 2022), exact cell indexing. "
            "td 2017-2020: TimesFM 3.0 predictions (research product, non-commercial "
            "weights, no observed truth). "
            + (
                f"prec: {Path(src.prec_nc_file).name} (PISCOp v3.0, Gutierrez et al. 2026) "
                f"interpolated bilinearly at the points, nan_policy={src.prec_nan_policy}."
                if src.prec_nc_file is not None
                else "prec: PISCOp v2 resampled bilinearly to 0.01 deg, extracted earlier with masking.py."
            )
            + (
                f" Variables {list(src.link_variables)} are symlinks to {src.link_from}."
                if src.link_variables
                else ""
            )
        ),
    }
    with open(base / "provenance.json", "w") as fh:
        json.dump(prov, fh, indent=2, default=str)

    for v, r in report.items():
        gaps = r["missing"]
        gap_txt = f"{len(gaps)} missing months (first: {gaps[0]})" if gaps else "complete"
        print(f"[prepare] {v}: {r['n_present']} months, {r['first']} .. {r['last']}, {gap_txt}")
    if failures:
        print(f"[prepare] {len(failures)} jobs failed")
    return report


if __name__ == "__main__":
    import argparse

    from config import EXAMPLE_SOURCES

    ap = argparse.ArgumentParser(description="Build SIMCAST input parquets")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--variables", nargs="+", default=["prec", "tmax", "tmin", "td"]
    )
    a = ap.parse_args()
    prepare_all(
        EXAMPLE_SOURCES,
        workers=a.workers,
        overwrite=a.overwrite,
        variables=tuple(a.variables),
    )
