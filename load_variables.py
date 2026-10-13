import glob
import json
import os
from collections import defaultdict
from datetime import datetime
from functools import lru_cache, reduce
from pathlib import Path

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from dask.distributed import Client, as_completed, progress
from rasterio.transform import from_origin

from point_attrs import add_point_columns, point_attributes
from simcast import calculate_hhr, simcast_model

DEFAULT_VARIABLES = ["prec", "tmax", "tmin", "td"]
DEFAULT_MAX_DATA_YEAR = 2020

# Days dropped by calculate_hhr at each end of the window.
_HHR_EDGE_DAYS = 2


def find_parquet_files(
    variable: str, date_range: tuple, *, base_path: str | os.PathLike[str]
) -> list[str]:
    """Find monthly files `<base>/<var>/Outputs/<var>_daily_YYYY_MM.parquet`."""
    base_path = os.fspath(base_path)
    start_date, end_date = pd.to_datetime(date_range)

    months_needed = (
        pd.date_range(start_date.replace(day=1), end_date, freq="MS")
        .strftime("%Y_%m")
        .unique()
    )

    files = []
    for ym in months_needed:
        pattern = os.path.join(
            base_path, variable, "Outputs", f"{variable}_daily_{ym}.parquet"
        )
        files.extend(glob.glob(pattern))

    return sorted(files)


def load_variable_data(
    variable: str,
    date_range: tuple,
    region_ids,
    *,
    base_path: str | os.PathLike[str],
) -> pd.DataFrame:
    """
    Load one variable for a set of IDs and a date window.

    The ID range filters let PyArrow skip whole row groups when the parquet is
    sorted by ID (the layout produced by `prepare_inputs.py`); the `in` filter
    then keeps only the requested IDs.
    """
    files = find_parquet_files(variable, date_range, base_path=base_path)
    if not files:
        return pd.DataFrame()

    ids = np.unique(np.asarray(region_ids))
    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[1])

    filters = [
        ("FECHA", ">=", start_date),
        ("FECHA", "<=", end_date),
        ("ID", ">=", int(ids.min())),
        ("ID", "<=", int(ids.max())),
        ("ID", "in", ids.tolist()),
    ]

    df = pd.read_parquet(
        files,
        filters=filters,
        columns=["ID", "FECHA", "Value"],
        engine="pyarrow",
    )
    if variable == "prec":
        df["Value"] = df["Value"].fillna(0.0)

    df["FECHA"] = pd.to_datetime(df["FECHA"]).dt.normalize()
    df["ID"] = df["ID"].astype(np.int64)

    return df.rename(columns={"Value": variable.upper()})


def load_climate_window(
    ids,
    date_range: tuple,
    *,
    base_path: str | os.PathLike[str],
    variables: list[str] = DEFAULT_VARIABLES,
) -> pd.DataFrame:
    """Merge all variables (inner join on ID, FECHA) into [ID, FECHA, PP, TMAX, TMIN, TDEW]."""
    dfs = []
    for var in variables:
        df = load_variable_data(var, date_range, ids, base_path=base_path)
        if df.empty:
            return pd.DataFrame()
        dfs.append(df)

    merged = reduce(
        lambda left, right: pd.merge(left, right, on=["ID", "FECHA"], how="inner"), dfs
    )
    if merged.empty:
        return merged

    merged = merged.rename(
        columns={"PREC": "PP", "TMAX": "TMAX", "TMIN": "TMIN", "TD": "TDEW"}
    )
    missing = {"PP", "TMAX", "TMIN", "TDEW"} - set(merged.columns)
    if missing:
        raise ValueError(
            f"Merged climate dataframe is missing required columns: {sorted(missing)}"
        )
    return merged.sort_values(["ID", "FECHA"], kind="stable").reset_index(drop=True)


@dask.delayed
def process_id_chunk(
    points: pd.DataFrame,
    date_range: tuple,
    vt: str,
    *,
    base_path: str | os.PathLike[str],
    variables: list[str] = DEFAULT_VARIABLES,
    rh_thresh: float = 90.0,
    timezone: str = "America/Lima",
    min_day: int = 5,
    forced_day: int = 25,
    sun_method: str = "noaa",
) -> pd.DataFrame:
    """
    Run HHR + SIMCAST for a chunk of grid points over one season.

    `points` has the columns [ID, lon, lat].  Returns the daily SIMCAST table
    for every ID that had complete climate data (may be empty).
    """
    merged = load_climate_window(
        points["ID"].to_numpy(), date_range, base_path=base_path, variables=variables
    )
    if merged.empty:
        return pd.DataFrame()

    lon = dict(zip(points["ID"].to_numpy(), points["lon"].to_numpy()))
    lat = dict(zip(points["ID"].to_numpy(), points["lat"].to_numpy()))

    results = []
    for grid_id, group in merged.groupby("ID", sort=False):
        try:
            hhr_df = calculate_hhr(
                group,
                lon=lon[grid_id],
                lat=lat[grid_id],
                rh_thresh=rh_thresh,
                timezone=timezone,
                sun_method=sun_method,
            )
            if hhr_df.empty:
                continue
            res = simcast_model(hhr_df, vt, min_day=min_day, forced_day=forced_day)
            res["ID"] = grid_id
            results.append(res)
        except Exception as e:  # keep the chunk alive; report the ID
            print(f"Error processing ID {grid_id}: {e}")

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


# Backwards-compatible name used by the README / notebook.
def process_region_period(
    region_gdf: gpd.GeoDataFrame, date_range: tuple, vt: str, **kwargs
) -> pd.DataFrame:
    """Process a whole region in one call (no Dask).  Prefer `parallel_simcast_pipeline`."""
    kwargs.pop("batch_size", None)
    pts = points_table(region_gdf)
    return process_id_chunk(pts, date_range, vt, **kwargs).compute(scheduler="sync")


def points_table(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """[ID, lon, lat] for a point GeoDataFrame (small, cheap to ship to workers)."""
    geom = gdf.geometry
    geom = geom.apply(lambda s: s if s.geom_type == "Point" else s.centroid)
    return pd.DataFrame(
        {
            "ID": gdf["ID"].to_numpy().astype(np.int64),
            "lon": geom.x.to_numpy(),
            "lat": geom.y.to_numpy(),
        }
    ).sort_values("ID", kind="stable")


def season_periods(year: int) -> dict[str, tuple[str, str]]:
    return {
        "cultivo1": (f"{year}-11-01", f"{year + 1}-04-30"),
        "cultivo2": (f"{year}-07-01", f"{year}-12-31"),
        "cultivo3": (f"{year}-04-01", f"{year}-07-31"),
    }


def region_periods(year: int):
    """Return {region_name: {period_name: (start, end)}} valid for that year."""
    p = season_periods(year)
    return {
        "Sierra Alta": {"cultivo1": p["cultivo1"]},
        "Sierra Media": {"cultivo3": p["cultivo3"]},
        "Costa y Valle Interandino": {"cultivo2": p["cultivo2"]},
    }


def results_path(results_root, year: int, period: str, region: str) -> Path:
    return Path(os.fspath(results_root)) / str(year) / period / region


def parallel_simcast_pipeline(
    start_year: int,
    end_year: int,
    vt: str,
    *,
    base_path: str | os.PathLike[str],
    potato_grid_file: str | os.PathLike[str],
    results_root: str | os.PathLike[str],
    variables: list[str] = DEFAULT_VARIABLES,
    n_workers: int = 20,
    threads_per_worker: int = 1,
    memory_limit: str | None = "5GB",
    dashboard_address: str = ":8787",
    client: Client | None = None,
    scheduler_address: str | None = None,
    write_tiff: bool = False,
    show_task_bar: bool = True,
    max_data_year: int = DEFAULT_MAX_DATA_YEAR,
    regions: dict[str, gpd.GeoDataFrame] | None = None,
    valid_pairs: set[tuple[str, str]] | None = None,
    id_chunk_size: int = 5_000,
    skip_existing: bool = True,
    batch_size: int | None = None,  # deprecated, ignored
    rh_thresh: float = 90.0,
    timezone: str = "America/Lima",
    min_day: int = 5,
    forced_day: int = 25,
    sun_method: str = "noaa",
):
    """
    Run SIMCAST over (year, region, period) seasons with Dask.

    Every season is split into chunks of `id_chunk_size` points; each chunk is
    an independent Dask task that reads only its own IDs from the monthly
    parquets, so peak RAM per task is small and all cores stay busy even
    though there are only three seasons per year.

    Outputs (per season): `<results_root>/<year>/<period>/<region>/simcast_results.parquet`
    plus `coverage.json` (requested vs. actual date range; a truncated season
    means the climate inputs end before the season does, e.g. prec in 2020).

    Resource tips:
      - `n_workers` x `threads_per_worker` = concurrent chunks. Keep
        `threads_per_worker=1`: the per-point work is pandas/Python bound and
        threads in one process serialise on the GIL (10x2 threads took 5x
        longer than 20x1 for the same year).
      - `memory_limit` is per worker process; ~0.5 GB per concurrent chunk of 5k IDs.
      - `skip_existing=True` makes reruns resume where they stopped.
      - Pass `client=` or `scheduler_address=` to use an existing / multi-machine cluster.
    """
    own_client = client is None
    if client is None:
        if scheduler_address:
            client = Client(scheduler_address)
        else:
            client = Client(
                n_workers=n_workers,
                threads_per_worker=threads_per_worker,
                memory_limit=memory_limit,
                dashboard_address=dashboard_address,
            )
    print("Dask dashboard:", client.dashboard_link)

    potato_grid = gpd.read_file(os.fspath(potato_grid_file))
    if "ID" not in potato_grid.columns:
        potato_grid["ID"] = potato_grid.index

    if regions is None:
        regions = split_regions_by_type(potato_grid)
    if valid_pairs is None:
        valid_pairs = default_valid_pairs()

    region_points = {name: points_table(gdf) for name, gdf in regions.items()}
    attrs = point_attributes(potato_grid)  # lon, lat, altitude written with the results

    futures_map: dict = {}
    n_chunks: dict[tuple, int] = {}
    skipped = []

    for y in range(start_year, end_year + 1):
        for period, dr in season_periods(y).items():
            for region, pts in region_points.items():
                if (region, period) not in valid_pairs:
                    continue
                if datetime.fromisoformat(dr[1]).year > max_data_year:
                    continue
                key = (y, period, region)
                out_dir = results_path(results_root, *key)
                if skip_existing and (out_dir / "simcast_results.parquet").exists():
                    skipped.append(key)
                    continue

                chunks = [
                    pts.iloc[i : i + id_chunk_size].reset_index(drop=True)
                    for i in range(0, len(pts), id_chunk_size)
                ]
                n_chunks[key] = len(chunks)
                for chunk in chunks:
                    fut = client.compute(
                        process_id_chunk(
                            chunk,
                            dr,
                            vt,
                            base_path=os.fspath(base_path),
                            variables=variables,
                            rh_thresh=rh_thresh,
                            timezone=timezone,
                            min_day=min_day,
                            forced_day=forced_day,
                            sun_method=sun_method,
                        )
                    )
                    futures_map[fut] = key

    if skipped:
        print(f"[simcast] skipping {len(skipped)} seasons with existing results")
    print(f"[simcast] {len(n_chunks)} seasons -> {len(futures_map)} chunk tasks")

    if show_task_bar and futures_map:
        progress(list(futures_map.keys()))

    parts: dict[tuple, list] = defaultdict(list)
    done: dict[tuple, int] = defaultdict(int)

    for fut, df in as_completed(futures_map, with_results=True):
        key = futures_map[fut]
        done[key] += 1
        if df is not None and not df.empty:
            parts[key].append(df)
        fut.release()

        if done[key] == n_chunks[key]:
            year, period, region = key
            season = season_periods(year)[period]
            if parts[key]:
                df_all = pd.concat(parts[key], ignore_index=True)
            else:
                df_all = pd.DataFrame()
            del parts[key]
            save_results(
                df_all,
                year,
                period,
                region,
                results_root=results_root,
                potato_grid_file=potato_grid_file,
                write_tiff=write_tiff,
                requested_range=season,
                n_points=len(region_points[region]),
                point_attrs=attrs,
            )
            del df_all

    if own_client:
        client.close()


@lru_cache(maxsize=4)
def _raster_grid(potato_grid_file: str):
    """Grid geometry shared by every raster written from the same point file."""
    potato_grid = gpd.read_file(potato_grid_file)
    if "ID" not in potato_grid.columns:
        potato_grid["ID"] = potato_grid.index

    x_coords = potato_grid.geometry.x.values
    y_coords = potato_grid.geometry.y.values

    dx = np.abs(np.diff(np.sort(np.unique(x_coords)))).min()
    dy = np.abs(np.diff(np.sort(np.unique(y_coords)))).min()

    transform = from_origin(x_coords.min() - dx / 2, y_coords.max() + dy / 2, dx, dy)
    # Same expressions as the original notebook so raster shapes stay identical.
    ncols = int((x_coords.max() - x_coords.min()) / dx) + 1
    nrows = int((y_coords.max() - y_coords.min()) / dy) + 1

    x_idx = ((x_coords - transform.c) / transform.a).astype(int)
    y_idx = ((y_coords - transform.f) / transform.e).astype(int)

    return {
        "ids": potato_grid["ID"].to_numpy(),
        "crs": potato_grid.crs,
        "transform": transform,
        "nrows": nrows,
        "ncols": ncols,
        "x_idx": x_idx,
        "y_idx": y_idx,
    }


def create_raster(
    result_df: pd.DataFrame,
    output_path: str | os.PathLike[str],
    *,
    potato_grid_file: str | os.PathLike[str],
    value_col: str = "APP",
    agg: str = "max",
):
    """
    Convert simulation results to a GeoTIFF (one band: `agg` of `value_col` per ID).
    """
    g = _raster_grid(os.fspath(potato_grid_file))
    raster = np.full((g["nrows"], g["ncols"]), -9999, dtype=np.int16)

    per_id = result_df.groupby("ID")[value_col].agg(agg)
    vals = per_id.reindex(g["ids"]).to_numpy()
    ok = ~np.isnan(vals)
    raster[g["y_idx"][ok], g["x_idx"][ok]] = vals[ok].astype(np.int16)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=g["nrows"],
        width=g["ncols"],
        count=1,
        dtype=raster.dtype,
        crs=g["crs"],
        transform=g["transform"],
        nodata=-9999,
    ) as dst:
        dst.write(raster, 1)


def save_results(
    df: pd.DataFrame,
    year: int,
    period: str,
    region: str,
    *,
    results_root: str | os.PathLike[str],
    potato_grid_file: str | os.PathLike[str],
    write_tiff: bool = False,
    requested_range: tuple[str, str] | None = None,
    n_points: int | None = None,
    point_attrs: pd.DataFrame | None = None,
):
    """
    Write ``simcast_results.parquet`` (+ ``coverage.json`` and optional APP tif).

    With ``point_attrs`` (see ``point_attrs.point_attributes``) the parquet also
    carries ``lon``, ``lat`` and ``altitude`` right after ``ID``.
    """
    out_dir = results_path(results_root, year, period, region)
    out_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        print(f"[save_results] {year}-{period}-{region}: no results (missing inputs?)")
        _write_coverage(out_dir, df, requested_range, n_points)
        return

    df = df.sort_values(["ID", "date"], kind="stable").reset_index(drop=True)
    if point_attrs is not None:
        df = add_point_columns(df, point_attrs)
    df.to_parquet(out_dir / "simcast_results.parquet")
    _write_coverage(out_dir, df, requested_range, n_points)

    if not write_tiff:
        return
    try:
        create_raster(
            df,
            out_dir / f"simcast_{year}_{period}_{region}.tif",
            potato_grid_file=potato_grid_file,
        )
    except Exception as e:
        print(f"[save_results] raster failed {year}-{period}-{region}: {e}")


def _write_coverage(out_dir: Path, df: pd.DataFrame, requested_range, n_points):
    info: dict = {"requested": list(requested_range) if requested_range else None}
    if df.empty:
        info.update({"first": None, "last": None, "n_days": 0, "n_ids": 0, "truncated": True})
    else:
        first, last = df["date"].min(), df["date"].max()
        info.update(
            {
                "first": str(first.date()),
                "last": str(last.date()),
                "n_days": int(df["date"].nunique()),
                "n_ids": int(df["ID"].nunique()),
                "n_points": n_points,
            }
        )
        if requested_range:
            exp_last = pd.Timestamp(requested_range[1]) - pd.Timedelta(days=_HHR_EDGE_DAYS)
            exp_first = pd.Timestamp(requested_range[0]) + pd.Timedelta(days=_HHR_EDGE_DAYS)
            info["truncated"] = bool(last < exp_last or first > exp_first)
            if info["truncated"]:
                print(
                    f"[save_results] WARNING {out_dir}: season truncated "
                    f"({first.date()} .. {last.date()}, requested {requested_range[0]} .. {requested_range[1]})"
                )
        if n_points and info["n_ids"] < n_points:
            print(
                f"[save_results] WARNING {out_dir}: {n_points - info['n_ids']} of "
                f"{n_points} points produced no result"
            )
    with open(out_dir / "coverage.json", "w") as fh:
        json.dump(info, fh, indent=2)


def split_regions_by_type(potato_grid: gpd.GeoDataFrame) -> dict[str, gpd.GeoDataFrame]:
    """
    Default Peru-specific split used in the original notebook.

    Requires a 'Type' column in the potato grid.
    """
    if "Type" not in potato_grid.columns:
        raise ValueError(
            "potato_grid is missing required column 'Type' for default region split"
        )

    return {
        "Sierra Alta": potato_grid[potato_grid["Type"] == "Sierra Alta [>3,000]"],
        "Sierra Media": potato_grid[
            potato_grid["Type"] == "Sierra Media [2000-3000 masl]"
        ],
        "Costa y Valle Interandino": potato_grid[
            potato_grid["Type"].isin(
                ["Costa [0-500 msnm]", "Interandino [500 - 2,000]"]
            )
        ],
    }


def default_valid_pairs() -> set[tuple[str, str]]:
    """Default (region, period) combinations allowed by the original notebook."""
    return {
        ("Sierra Alta", "cultivo1"),
        ("Sierra Media", "cultivo3"),
        ("Costa y Valle Interandino", "cultivo2"),
    }
