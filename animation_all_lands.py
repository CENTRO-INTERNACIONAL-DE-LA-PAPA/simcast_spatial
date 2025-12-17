import glob
import logging
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

logger = logging.getLogger(__name__)


DEFAULT_GADM_PERU_GPKG = "https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_PER.gpkg"
DEFAULT_CMAP = "PuRd"
DEFAULT_ALPHA = 0.65
DEFAULT_CULTIVOS = {
    "cultivo1": "Sierra Alta",
    "cultivo2": "Costa y Valle Interandino",
    "cultivo3": "Sierra Media",
}


def load_gadm_peru(gadm_gpkg: str | os.PathLike[str] = DEFAULT_GADM_PERU_GPKG):
    """Load Peru admin-0 and admin-1 boundaries from a GADM GeoPackage (URL or local path)."""
    gadm_admin1 = gpd.read_file(gadm_gpkg, layer="ADM_ADM_1")  # admin-1
    gadm_admin0 = gpd.read_file(gadm_gpkg, layer="ADM_ADM_0")  # admin-0
    return gadm_admin0, gadm_admin1


def create_raster(
    result_df: pd.DataFrame,
    output_path: str | os.PathLike[str],
    value_column: str,
    agg_method: str,
    *,
    grid_file_path: str | os.PathLike[str],
):
    """Convert aggregated column values from a results dataframe into a GeoTIFF raster."""
    try:
        logger.info(
            "Creating raster for '%s' (agg: %s) -> %s",
            value_column,
            agg_method,
            os.path.basename(str(output_path)),
        )
        potato_grid = gpd.read_file(os.fspath(grid_file_path))
        if "ID" not in potato_grid.columns:
            potato_grid["ID"] = potato_grid.index

        agg_df = (
            result_df.groupby("ID")
            .agg(agg_value=(value_column, agg_method))
            .reset_index()
            .rename(columns={"agg_value": value_column})
        )

        merged = potato_grid.merge(agg_df, on="ID", how="left")

        grid_crs = potato_grid.crs
        x_coords, y_coords = merged.geometry.x, merged.geometry.y
        dx = np.abs(np.diff(np.sort(x_coords.unique()))).min()
        dy = np.abs(np.diff(np.sort(y_coords.unique()))).min()
        transform = from_origin(
            x_coords.min() - dx / 2, y_coords.max() + dy / 2, dx, dy
        )

        ncols = int((x_coords.max() - x_coords.min()) / dx) + 1
        nrows = int((y_coords.max() - y_coords.min()) / dy) + 1
        raster = np.full((nrows, ncols), -9999.0, dtype=np.float32)

        x_idx_map = {
            int(row["ID"]): int((row.geometry.x - transform.c) / transform.a)
            for _, row in merged.iterrows()
            if pd.notna(row.geometry.x)
        }
        y_idx_map = {
            int(row["ID"]): int((row.geometry.y - transform.f) / transform.e)
            for _, row in merged.iterrows()
            if pd.notna(row.geometry.y)
        }

        for _, row in merged.iterrows():
            if pd.notna(row[value_column]):
                row_id = int(row["ID"])
                if row_id in x_idx_map and row_id in y_idx_map:
                    raster[y_idx_map[row_id], x_idx_map[row_id]] = row[value_column]

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
            nodata=-9999.0,
        ) as dst:
            dst.write(raster, 1)

    except Exception as e:
        logger.error(
            "FAILED to create raster for '%s': %s", value_column, e, exc_info=True
        )


def generate_bu_fu_rasters(
    *,
    results_root: str | os.PathLike[str],
    grid_file_path: str | os.PathLike[str],
    parquet_name: str = "simcast_results.parquet",
):
    """
    Finds all existing simcast_results.parquet files and generates FU and BU rasters for them.

    Produces (for each parquet):
      - FU_sum_<year>_<period>_<region>.tif
      - BU_sum_<year>_<period>_<region>.tif
    """
    logger.info("--- Starting raster creation from existing parquet files ---")

    search_pattern = os.path.join(os.fspath(results_root), "**", parquet_name)
    all_parquet_files = glob.glob(search_pattern, recursive=True)

    if not all_parquet_files:
        logger.error(
            "No '%s' files found in '%s'.", parquet_name, os.fspath(results_root)
        )
        return

    logger.info("Found %d parquet files to process.", len(all_parquet_files))

    for parquet_file in all_parquet_files:
        try:
            logger.info("Processing: %s", parquet_file)
            output_dir = Path(parquet_file).parent

            parts = output_dir.parts
            region = parts[-1]
            period = parts[-2]
            year = parts[-3]

            df = pd.read_parquet(parquet_file)
            if df.empty:
                logger.warning("Skipping empty parquet file.")
                continue

            if "FU" in df.columns:
                create_raster(
                    df,
                    output_path=output_dir / f"FU_sum_{year}_{period}_{region}.tif",
                    value_column="FU",
                    agg_method="sum",
                    grid_file_path=grid_file_path,
                )
            else:
                logger.warning("'FU' column not found. Skipping FU raster.")

            if "BU" in df.columns:
                create_raster(
                    df,
                    output_path=output_dir / f"BU_sum_{year}_{period}_{region}.tif",
                    value_column="BU",
                    agg_method="sum",
                    grid_file_path=grid_file_path,
                )
            else:
                logger.warning("'BU' column not found. Skipping BU raster.")

        except Exception as e:
            logger.error("Error while processing %s: %s", parquet_file, e)
            continue

    logger.info("--- Raster creation finished ---")


def paths_for_year(
    year: int,
    *,
    results_root: str | os.PathLike[str],
    cultivos: dict[str, str] = DEFAULT_CULTIVOS,
    raster_glob: str = "BU*.tif*",
    strict: bool = True,
) -> dict[str, str | None] | None:
    """Return {cultivo_key: path|None} for a year; returns None if the year directory doesn't exist."""
    year_dir = Path(os.fspath(results_root)) / str(year)
    if not year_dir.is_dir():
        return None

    paths = {}
    for cultivo_key, region in cultivos.items():
        pattern = str(year_dir / cultivo_key / region / raster_glob)
        matches = glob.glob(pattern)

        if not matches and strict:
            raise FileNotFoundError(f"Missing raster for {year}/{cultivo_key}")
        if len(matches) > 1:
            raise RuntimeError(f"Multiple rasters for {year}/{cultivo_key}")
        paths[cultivo_key] = matches[0] if matches else None
    return paths


def collect_paths_by_year(
    *,
    results_root: str | os.PathLike[str],
    start_year: int,
    end_year: int,
    cultivos: dict[str, str] = DEFAULT_CULTIVOS,
    raster_glob: str = "BU*.tif*",
    strict: bool = True,
) -> dict[int, dict[str, str | None]]:
    paths_by_year: dict[int, dict[str, str | None]] = {}
    for yr in range(start_year, end_year + 1):
        try:
            yr_paths = paths_for_year(
                yr,
                results_root=results_root,
                cultivos=cultivos,
                raster_glob=raster_glob,
                strict=strict,
            )
            if yr_paths:
                paths_by_year[yr] = yr_paths
        except Exception as exc:
            print(f"Skipping {yr}: {exc}")
    if not paths_by_year:
        raise RuntimeError("No valid years found!")
    return paths_by_year


def build_all_lands_data(paths_by_year: dict[int, dict[str, str | None]]):
    """Read BU rasters for each year/cultivo and build the combined masked arrays + extent/CRS."""
    sample_path = next(p for p in paths_by_year[list(paths_by_year)[0]].values() if p)
    with rasterio.open(sample_path) as ref:
        ref_shape = ref.shape
        ref_transform = ref.transform
        ref_crs = ref.crs
        left, top = ref_transform * (0, 0)
        right, bottom = ref_transform * (ref.width, ref.height)
        extent = [left, right, bottom, top]

    empty_layer = np.ma.masked_all(ref_shape)
    cultivo_keys = list(next(iter(paths_by_year.values())).keys())

    all_data = []
    for yr in sorted(paths_by_year):
        cultivo_arrays = []
        for cultivo in cultivo_keys:
            p = paths_by_year[yr].get(cultivo)
            if p is None:
                arr = empty_layer
            else:
                with rasterio.open(p) as src:
                    if (src.shape, src.transform) != (ref_shape, ref_transform):
                        raise ValueError(f"Grid mismatch in {p}")
                    arr = src.read(1)
                    arr = np.ma.masked_where(arr == -9999, arr)
            cultivo_arrays.append(arr)

        summed = np.zeros(ref_shape, dtype=np.float32)
        combined_mask = None
        for arr in cultivo_arrays:
            summed += arr.filled(0).astype(np.float32, copy=False)
            combined_mask = (
                arr.mask if combined_mask is None else (combined_mask & arr.mask)
            )
        combined_arr = np.ma.masked_array(summed, mask=combined_mask)
        all_data.append(combined_arr)

    stack = np.ma.stack(all_data)
    vmin, vmax = stack.min(), stack.max()

    return all_data, extent, ref_crs, vmin, vmax, empty_layer


def create_all_lands_animation(
    *,
    all_data,
    extent,
    years,
    gadm_admin0=None,
    gadm_admin1=None,
    ref_crs=None,
    vmin=None,
    vmax=None,
    cmap: str = DEFAULT_CMAP,
    alpha: float = DEFAULT_ALPHA,
    interval: int = 400,
    colorbar_label: str = "Blight Units",
):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if ref_crs is not None and gadm_admin0 is not None and gadm_admin1 is not None:
        gadm_admin0 = gadm_admin0.to_crs(ref_crs)
        gadm_admin1 = gadm_admin1.to_crs(ref_crs)

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(bottom=0.1)

    im = ax.imshow(
        np.ma.masked_all(all_data[0].shape),
        extent=extent,
        origin="upper",
        cmap=cmap,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
    )

    if gadm_admin0 is not None:
        gadm_admin0.boundary.plot(ax=ax, edgecolor="black", linewidth=1.2, zorder=10)
    if gadm_admin1 is not None:
        gadm_admin1.boundary.plot(ax=ax, edgecolor="dimgray", linewidth=0.8, zorder=9)

    title = ax.text(0.5, 1.05, "", transform=ax.transAxes, ha="center", fontsize=14)

    cbar = fig.colorbar(im)
    cbar.set_label(colorbar_label, labelpad=2)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, ls="--", alpha=0.4)

    def init():
        im.set_data(np.ma.masked_all(all_data[0].shape))
        title.set_text("")
        return [im, title]

    def update(frame):
        im.set_data(all_data[frame])
        title.set_text(f"SIMCAST Results – Year: {years[frame]}")
        return [im, title]

    anim = FuncAnimation(
        fig, update, frames=len(all_data), init_func=init, blit=True, interval=interval
    )
    return fig, ax, anim


def save_all_lands_animation(
    anim,
    *,
    gif_path: str | None = None,
    mp4_path: str | None = None,
    fps: int = 3,
    dpi: int = 150,
):
    from matplotlib.animation import FFMpegWriter, PillowWriter

    if gif_path is not None:
        anim.save(gif_path, writer=PillowWriter(fps=fps))
    if mp4_path is not None:
        writer = FFMpegWriter(fps=fps, bitrate=1800)
        anim.save(mp4_path, writer=writer, dpi=dpi)
