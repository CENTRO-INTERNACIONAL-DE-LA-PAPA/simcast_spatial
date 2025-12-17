import concurrent.futures
import glob
import os
from itertools import repeat

import geopandas as gpd
import pandas as pd
import xarray as xr
from tqdm import tqdm


def load_potato_grid(
    potato_grid_file: str | os.PathLike[str], *, id_col: str = "ID"
) -> gpd.GeoDataFrame:
    potato_grid_file = os.fspath(potato_grid_file)
    potato_grid = gpd.read_file(potato_grid_file)
    potato_grid["geometry"] = potato_grid.geometry.apply(
        lambda geom: geom.centroid if geom.geom_type != "Point" else geom
    )
    if id_col not in potato_grid.columns:
        potato_grid[id_col] = potato_grid.index
    return potato_grid


def extract_chunk(
    raster, gdf_chunk, time_name: str, id_col: str = "ID"
) -> pd.DataFrame:
    """
    Vectorized extraction on a chunk of points.

    Parameters:
        raster (xarray.DataArray): Raster DataArray with dimension "time" (after renaming) and spatial dims "y", "x".
        gdf_chunk (GeoDataFrame): A chunk (subset) of the points GeoDataFrame.
        time_name (str): Name of the time coordinate/dimension (e.g. "time").
        id_col (str): The column name for the point ID.

    Returns:
        pandas.DataFrame: A long-format DataFrame with columns [id_col, 'FECHA', 'Value'].
    """
    xs = gdf_chunk.geometry.x.values
    ys = gdf_chunk.geometry.y.values

    extracted = raster.interp(x=("points", xs), y=("points", ys), method="nearest")
    extracted = extracted.transpose("points", time_name)

    times = pd.to_datetime(extracted.coords[time_name].values)
    df_wide = pd.DataFrame(extracted.values, columns=times)

    if id_col in gdf_chunk.columns:
        df_wide.insert(0, id_col, gdf_chunk[id_col].values)
    else:
        df_wide.insert(0, id_col, gdf_chunk.index)

    return df_wide.melt(id_vars=[id_col], var_name="FECHA", value_name="Value")


def vectorized_extraction_with_chunks(
    raster, gdf, time_name, chunk_size: int = 10000, id_col: str = "ID"
) -> pd.DataFrame:
    """
    Splits the point GeoDataFrame into chunks and applies vectorized extraction on each chunk.
    """
    dfs = []
    n_points = len(gdf)
    for i in range(0, n_points, chunk_size):
        chunk = gdf.iloc[i : i + chunk_size]
        df_chunk = extract_chunk(raster, chunk, time_name, id_col)
        dfs.append(df_chunk)
    return pd.concat(dfs, ignore_index=True)


def process_file(
    file,
    potato_grid,
    var_name,
    time_name="T",
    chunk_size: int = 10000,
    id_col: str = "ID",
):
    ds = xr.open_dataset(file)
    if var_name in ds.data_vars:
        raster = ds[var_name]
    else:
        raster = list(ds.data_vars.values())[0]

    if "time" not in raster.coords and "T" in raster.coords:
        raster = raster.rename({"T": "time"})
    if "x" not in raster.coords and "X" in raster.coords:
        raster = raster.rename({"X": "x"})
    if "y" not in raster.coords and "Y" in raster.coords:
        raster = raster.rename({"Y": "y"})
    if "x" not in raster.coords and "longitude" in raster.coords:
        raster = raster.rename({"longitude": "x", "latitude": "y"})

    extracted_df = vectorized_extraction_with_chunks(
        raster, potato_grid, time_name, chunk_size, id_col
    )
    extracted_df["source_file"] = os.path.basename(file)
    return extracted_df


def extract_folder_to_parquets(
    *,
    resample_folder: str | os.PathLike[str],
    output_folder: str | os.PathLike[str],
    potato_grid_file: str | os.PathLike[str],
    glob_pattern: str,
    var_name: str,
    time_name: str = "time",
    chunk_size: int = 10000,
    id_col: str = "ID",
    max_workers: int | None = None,
):
    """
    Extract point values from a folder of NetCDF rasters and save monthly Parquet files.

    Assumes each NetCDF file contains a time dimension/coord and spatial coords compatible
    with the potato grid geometry.

    Resource tips:
      - `chunk_size` controls peak RAM inside each worker (smaller = less RAM, slower).
      - `max_workers` controls parallelism (more workers = faster, more RAM/I/O).
    """
    resample_folder = os.fspath(resample_folder)
    output_folder = os.fspath(output_folder)
    potato_grid = load_potato_grid(potato_grid_file, id_col=id_col)

    os.makedirs(output_folder, exist_ok=True)
    files = glob.glob(os.path.join(resample_folder, glob_pattern))
    print(f"Found {len(files)} files.")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for df in tqdm(
            executor.map(
                process_file,
                files,
                repeat(potato_grid),
                repeat(var_name),
                repeat(time_name),
                repeat(chunk_size),
                repeat(id_col),
            ),
            total=len(files),
            desc="Processing files",
        ):
            file_name = df["source_file"].iloc[0].split(".")[0]
            df.to_parquet(
                os.path.join(output_folder, f"{file_name}.parquet"), engine="pyarrow"
            )
            del df
