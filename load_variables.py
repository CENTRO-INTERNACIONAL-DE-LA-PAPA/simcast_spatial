import glob
import os
from datetime import datetime
from functools import reduce
from pathlib import Path

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from dask.distributed import Client, as_completed, progress
from rasterio.transform import from_origin

from simcast import calculate_hhr, simcast_model

DEFAULT_VARIABLES = ["prec", "tmax", "tmin", "td"]
DEFAULT_MAX_DATA_YEAR = 2020


def find_parquet_files(
    variable: str, date_range: tuple, *, base_path: str | os.PathLike[str]
) -> list[str]:
    """Find files using actual naming convention observed."""
    base_path = os.fspath(base_path)
    start_date, end_date = pd.to_datetime(date_range)

    # Generate YYYYMM format months
    months_needed = (
        pd.date_range(start_date, end_date, freq="MS").strftime("%Y_%m").unique()
    )

    files = []
    for ym in months_needed:
        pattern = os.path.join(
            base_path, variable, "Outputs", f"{variable}_daily_{ym}.parquet"
        )
        matches = glob.glob(pattern)
        files.extend(matches)

    return sorted(files)


def load_variable_data(
    variable: str,
    date_range: tuple,
    region_ids: list,
    *,
    base_path: str | os.PathLike[str],
) -> pd.DataFrame:
    """Load data with partition pruning and row-group filtering."""
    files = find_parquet_files(variable, date_range, base_path=base_path)
    if not files:
        return pd.DataFrame()

    # Convert dates to numpy datetime64 for filter compatibility
    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[1])

    # Build PyArrow filters for partition pruning and row-group filtering
    filters = [
        ("FECHA", ">=", start_date),
        ("FECHA", "<=", end_date),
        ("ID", "in", region_ids),
    ]

    df = pd.read_parquet(
        files,
        filters=filters,
        columns=["ID", "FECHA", "Value"],
        engine="pyarrow",
        dtype_backend="numpy_nullable",
    )
    if variable == "prec":
        df["Value"] = df["Value"].fillna(0.0)

    df["FECHA"] = pd.to_datetime(df["FECHA"]).dt.normalize()

    return df.rename(columns={"Value": variable.upper()})


@dask.delayed
def process_region_period(
    region_gdf: gpd.GeoDataFrame,
    date_range: tuple,
    vt: str,
    *,
    base_path: str | os.PathLike[str],
    variables: list[str] = DEFAULT_VARIABLES,
    batch_size: int = 300,
    rh_thresh: float = 90.0,
    timezone: str = "America/Bogota",
    min_day: int = 5,
    forced_day: int = 25,
) -> pd.DataFrame:
    """Process a region-period combination with monthly parquet files."""
    region_ids = region_gdf["ID"].tolist()

    dfs = []
    for var in variables:
        df = load_variable_data(var, date_range, region_ids, base_path=base_path)
        if df.empty:
            return pd.DataFrame()
        dfs.append(df)

    merged_df = reduce(
        lambda left, right: pd.merge(left, right, on=["ID", "FECHA"], how="inner"), dfs
    )
    if merged_df.empty:
        return pd.DataFrame()

    rename_map = {
        "PREC": "PP",
        "TMAX": "TMAX",
        "TMIN": "TMIN",
        "TD": "TDEW",
    }
    merged_df = merged_df.rename(columns=rename_map)

    missing = {"PP", "TMAX", "TMIN", "TDEW"} - set(merged_df.columns)
    if missing:
        raise ValueError(
            f"Merged climate dataframe is missing required columns: {sorted(missing)}"
        )

    # 2. Split into memory-managed batches
    unique_ids = merged_df["ID"].unique()
    batches = np.array_split(unique_ids, np.ceil(len(unique_ids) / batch_size))

    results = []
    for batch_ids in batches:
        try:
            batch_df = merged_df[merged_df["ID"].isin(batch_ids)]
            batch_gdf = region_gdf[region_gdf["ID"].isin(batch_ids)]

            batch_results = []
            for grid_id, group in batch_df.groupby("ID"):
                try:
                    geom = batch_gdf[batch_gdf["ID"] == grid_id].geometry.iloc[0]

                    hhr_df = calculate_hhr(
                        group,
                        lon=geom.x,
                        lat=geom.y,
                        rh_thresh=rh_thresh,
                        timezone=timezone,
                    )

                    simcast_result = simcast_model(
                        hhr_df, vt, min_day=min_day, forced_day=forced_day
                    )
                    simcast_result["ID"] = grid_id
                    batch_results.append(simcast_result)
                except Exception as e:
                    print(f"Error processing ID {grid_id}: {str(e)}")
                    continue

            del batch_df, batch_gdf
            if batch_results:
                results.append(pd.concat(batch_results))

        except Exception as batch_error:
            print(f"Batch failed: {str(batch_error)}")
            continue

    if not results:
        return pd.DataFrame()
    return pd.concat(results).reset_index(drop=True)


def region_periods(year: int):
    """Return {region_name: {period_name: (start, end)}} valid for that year."""
    return {
        "Sierra Alta": {"cultivo1": (f"{year}-11-01", f"{year + 1}-04-30")},
        "Sierra Media": {"cultivo3": (f"{year}-04-01", f"{year}-07-31")},
        "Costa y Valle Interandino": {"cultivo2": (f"{year}-07-01", f"{year}-12-31")},
    }


def parallel_simcast_pipeline(
    start_year: int,
    end_year: int,
    vt: str,
    *,
    base_path: str | os.PathLike[str],
    potato_grid_file: str | os.PathLike[str],
    results_root: str | os.PathLike[str],
    variables: list[str] = DEFAULT_VARIABLES,
    n_workers: int = 8,
    threads_per_worker: int = 4,
    memory_limit: str | None = "14GB",
    dashboard_address: str = ":8787",
    write_tiff: bool = False,
    show_task_bar: bool = True,
    max_data_year: int = DEFAULT_MAX_DATA_YEAR,
    regions: dict[str, gpd.GeoDataFrame] | None = None,
    valid_pairs: set[tuple[str, str]] | None = None,
    batch_size: int = 300,
    rh_thresh: float = 90.0,
    timezone: str = "America/Bogota",
    min_day: int = 5,
    forced_day: int = 25,
):
    """
    Run the SIMCAST pipeline over (year, region, period) tasks using Dask.

    Resource tips:
      - Reduce `n_workers` / `threads_per_worker` on smaller CPUs.
      - Reduce `batch_size` to lower peak RAM per task (at the cost of speed).
      - Set `memory_limit` to match your machine; Dask may spill/kill workers if exceeded.
    """
    client = Client(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        dashboard_address=dashboard_address,
    )
    print("Dask dashboard:", client.dashboard_link)

    potato_grid = gpd.read_file(os.fspath(potato_grid_file))
    potato_grid["ID"] = potato_grid.get("ID", potato_grid.index)

    _ = client.scatter(potato_grid, broadcast=True)

    if regions is None:
        regions = split_regions_by_type(potato_grid)

    if valid_pairs is None:
        valid_pairs = default_valid_pairs()

    futures_map = {}

    for y in range(start_year, end_year + 1):
        periods = {
            "cultivo1": (f"{y}-11-01", f"{y + 1}-04-30"),
            "cultivo2": (f"{y}-07-01", f"{y}-12-31"),
            "cultivo3": (f"{y}-04-01", f"{y}-07-31"),
        }

        for region, rdf in regions.items():
            for period, dr in periods.items():
                if (region, period) not in valid_pairs:
                    continue
                if datetime.fromisoformat(dr[1]).year > max_data_year:
                    continue

                fut = client.compute(
                    process_region_period(
                        rdf.copy(),
                        dr,
                        vt,
                        base_path=base_path,
                        variables=variables,
                        batch_size=batch_size,
                        rh_thresh=rh_thresh,
                        timezone=timezone,
                        min_day=min_day,
                        forced_day=forced_day,
                    )
                )
                futures_map[fut] = (y, period, region)

    if show_task_bar:
        progress(list(futures_map.keys()))

    for fut, df in as_completed(futures_map, with_results=True):
        year, period, region = futures_map[fut]
        if df.empty:
            continue
        save_results(
            df,
            year,
            period,
            region,
            results_root=results_root,
            potato_grid_file=potato_grid_file,
            write_tiff=write_tiff,
        )

    client.close()


def create_raster(
    result_df: pd.DataFrame,
    output_path: str | os.PathLike[str],
    *,
    potato_grid_file: str | os.PathLike[str],
):
    """
    Convert simulation results to GeoTIFF raster.

    Parameters:
        result_df: DataFrame with columns ['ID', 'APP', ...] (must contain 'APP' for applications)
        output_path: Path for output GeoTIFF file
    """
    potato_grid = gpd.read_file(os.fspath(potato_grid_file))
    if "ID" not in potato_grid.columns:
        potato_grid["ID"] = potato_grid.index

    grid_crs = potato_grid.crs
    x_coords = potato_grid.geometry.x.values
    y_coords = potato_grid.geometry.y.values

    dx = np.abs(np.diff(np.sort(np.unique(x_coords)))).min()
    dy = np.abs(np.diff(np.sort(np.unique(y_coords)))).min()

    transform = from_origin(x_coords.min() - dx / 2, y_coords.max() + dy / 2, dx, dy)

    ncols = int((x_coords.max() - x_coords.min()) / dx) + 1
    nrows = int((y_coords.max() - y_coords.min()) / dy) + 1
    raster = np.full((nrows, ncols), -9999, dtype=np.int16)

    x_idx = ((x_coords - transform.c) / transform.a).astype(int)
    y_idx = ((y_coords - transform.f) / transform.e).astype(int)

    merged = potato_grid.merge(
        result_df.groupby("ID")["APP"].max().reset_index(), on="ID", how="left"
    )

    for _, row in merged.iterrows():
        if not np.isnan(row["APP"]):
            raster[y_idx[row.name], x_idx[row.name]] = int(row["APP"])

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=nrows,
        width=ncols,
        count=1,
        dtype=raster.dtype,
        crs=grid_crs,
        transform=transform,
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
):
    out_dir = Path(os.fspath(results_root)) / str(year) / period / region
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "simcast_results.parquet")

    if not write_tiff or df.empty:
        return

    try:
        create_raster(
            df,
            out_dir / f"simcast_{year}_{period}_{region}.tif",
            potato_grid_file=potato_grid_file,
        )
    except Exception as e:
        print(f"[save_results] raster failed {year}-{period}-{region}: {e}")


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
