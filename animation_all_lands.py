import concurrent.futures as cf
import glob
import logging
import os
import urllib.request
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

logger = logging.getLogger(__name__)


DEFAULT_GADM_PERU_GPKG = "https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_PER.gpkg"
DEFAULT_GADM_CACHE_DIR = Path(
    os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
) / "simcast_spatial"
DEFAULT_CMAP = "PuRd"
DEFAULT_ALPHA = 0.65
# Rasters animated per variable: glob inside <year>/<cultivo>/<region>/, colorbar
# label and output file stem.  APP rasters are written by the SIMCAST step
# (max APP per point); BU/FU rasters by `generate_bu_fu_rasters` (season sums).
ANIMATION_VARIABLES = {
    "APP": {
        "raster_glob": "simcast_*.tif",
        "label": "Aplicaciones de fungicida por campaña (APP)",
        "stem": "lateblight_all_lands_app",
    },
    "BU": {
        "raster_glob": "BU_sum_*.tif",
        "label": "Blight Units, suma por campaña (BU)",
        "stem": "lateblight_all_lands_bu",
    },
    "FU": {
        "raster_glob": "FU_sum_*.tif",
        "label": "Fungicide Units, suma por campaña (FU)",
        "stem": "lateblight_all_lands_fu",
    },
}
DEFAULT_CULTIVOS = {
    "cultivo1": "Sierra Alta",
    "cultivo2": "Costa y Valle Interandino",
    "cultivo3": "Sierra Media",
}


def _download(url: str, dest: Path, *, retries: int = 3, timeout: int = 60) -> None:
    """Download ``url`` to ``dest`` with a browser-like User-Agent and retries."""
    import time

    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) simcast-spatial/0.1"}
    )
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as fh:
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
            return
        except Exception as e:  # HTTP 404/5xx or network hiccups
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not download {url}: {last}")


def load_gadm_peru(
    gadm_gpkg: str | os.PathLike[str] = DEFAULT_GADM_PERU_GPKG,
    *,
    cache_dir: str | os.PathLike[str] = DEFAULT_GADM_CACHE_DIR,
):
    """
    Load Peru admin-0 and admin-1 boundaries from a GADM GeoPackage.

    A URL is downloaded once into ``cache_dir`` (~28 MB) and read from there
    afterwards, instead of being fetched again by every map.
    """
    src = os.fspath(gadm_gpkg)
    if src.startswith(("http://", "https://")):
        cache_dir = Path(os.fspath(cache_dir))
        cache_dir.mkdir(parents=True, exist_ok=True)
        local = cache_dir / Path(src).name
        if not local.exists():
            logger.info("Downloading %s -> %s", src, local)
            tmp = local.with_suffix(".part")
            try:
                _download(src, tmp)
                os.replace(tmp, local)
                src = os.fspath(local)
            except Exception as e:  # fall back to GDAL's own remote reader
                logger.warning("Download failed (%s); reading %s directly", e, src)
                tmp.unlink(missing_ok=True)
        else:
            src = os.fspath(local)
    gadm_admin1 = gpd.read_file(src, layer="ADM_ADM_1")  # admin-1
    gadm_admin0 = gpd.read_file(src, layer="ADM_ADM_0")  # admin-0
    return gadm_admin0, gadm_admin1


def create_raster(
    result_df: pd.DataFrame,
    output_path: str | os.PathLike[str],
    value_column: str,
    agg_method: str,
    *,
    grid_file_path: str | os.PathLike[str],
):
    """Convert aggregated column values from a results dataframe into a GeoTIFF raster (float32)."""
    from load_variables import _raster_grid

    logger.info(
        "Creating raster for '%s' (agg: %s) -> %s",
        value_column,
        agg_method,
        os.path.basename(str(output_path)),
    )
    g = _raster_grid(os.fspath(grid_file_path))
    raster = np.full((g["nrows"], g["ncols"]), -9999.0, dtype=np.float32)

    per_id = result_df.groupby("ID")[value_column].agg(agg_method)
    vals = per_id.reindex(g["ids"]).to_numpy(dtype=np.float64)
    ok = ~np.isnan(vals)
    raster[g["y_idx"][ok], g["x_idx"][ok]] = vals[ok].astype(np.float32)

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
        nodata=-9999.0,
    ) as dst:
        dst.write(raster, 1)


def _rasters_for_file(
    parquet_file: str,
    grid_file_path: str,
    *,
    columns: tuple[str, ...] = ("FU", "BU"),
    skip_existing: bool = True,
) -> list[str]:
    output_dir = Path(parquet_file).parent
    region, period, year = output_dir.parts[-1], output_dir.parts[-2], output_dir.parts[-3]
    targets = {
        col: output_dir / f"{col}_sum_{year}_{period}_{region}.tif" for col in columns
    }
    todo = {c: p for c, p in targets.items() if not (skip_existing and p.exists())}
    if not todo:
        return []

    df = pd.read_parquet(parquet_file, columns=["ID", *todo])
    if df.empty:
        logger.warning("Skipping empty parquet file %s", parquet_file)
        return []
    written = []
    for col, out in todo.items():
        create_raster(df, out, value_column=col, agg_method="sum", grid_file_path=grid_file_path)
        written.append(os.fspath(out))
    return written


def generate_bu_fu_rasters(
    *,
    results_root: str | os.PathLike[str],
    grid_file_path: str | os.PathLike[str],
    parquet_name: str = "simcast_results.parquet",
    workers: int = 8,
    skip_existing: bool = True,
) -> list[str]:
    """
    Finds all existing simcast_results.parquet files and generates FU and BU rasters for them.

    Produces (for each parquet):
      - FU_sum_<year>_<period>_<region>.tif
      - BU_sum_<year>_<period>_<region>.tif

    Files are processed in parallel (``workers`` processes); rasters that
    already exist are skipped unless ``skip_existing=False``.
    """
    logger.info("--- Starting raster creation from existing parquet files ---")

    search_pattern = os.path.join(os.fspath(results_root), "*", "*", "*", parquet_name)
    all_parquet_files = sorted(glob.glob(search_pattern))
    if not all_parquet_files:
        logger.error("No '%s' files found in '%s'.", parquet_name, os.fspath(results_root))
        return []
    logger.info("Found %d parquet files to process.", len(all_parquet_files))

    written: list[str] = []
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(
                _rasters_for_file, f, os.fspath(grid_file_path), skip_existing=skip_existing
            ): f
            for f in all_parquet_files
        }
        for fut in cf.as_completed(futs):
            try:
                written.extend(fut.result())
            except Exception as e:
                logger.error("Error while processing %s: %s", futs[fut], e)

    logger.info("--- Raster creation finished (%d rasters written) ---", len(written))
    return written


def paths_for_year(
    year: int,
    *,
    results_root: str | os.PathLike[str],
    cultivos: dict[str, str] = DEFAULT_CULTIVOS,
    raster_glob: str = "BU*.tif*",
    strict: bool = True,
) -> dict[str, str | None] | None:
    """
    Return {cultivo_key: path|None} for a year.

    Returns None if the year directory doesn't exist or (with ``strict=False``)
    no cultivo has a raster at all.
    """
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
    if all(p is None for p in paths.values()):
        return None
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
            else:
                print(f"Skipping {yr}: no rasters")
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
                    nodata = src.nodata if src.nodata is not None else -9999
                    arr = src.read(1)
                    arr = np.ma.masked_where(arr == nodata, arr)
            cultivo_arrays.append(arr)

        summed = np.zeros(ref_shape, dtype=np.float32)
        combined_mask = None
        for arr in cultivo_arrays:
            summed += arr.filled(0).astype(np.float32, copy=False)
            mask = np.ma.getmaskarray(arr)
            combined_mask = mask if combined_mask is None else (combined_mask & mask)
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
    title_prefix: str = "SIMCAST Results",
    colorbar_extend: str = "neither",
    classes=None,
):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if ref_crs is not None and gadm_admin0 is not None and gadm_admin1 is not None:
        gadm_admin0 = gadm_admin0.to_crs(ref_crs)
        gadm_admin1 = gadm_admin1.to_crs(ref_crs)

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(bottom=0.16 if classes is not None else 0.1)

    if classes is not None:
        # discrete classes: one colour per class, opaque so steps stay distinct
        cmap_c, norm_c = classes.cmap_norm()
        im = ax.imshow(
            np.ma.masked_all(all_data[0].shape),
            extent=extent,
            origin="upper",
            cmap=cmap_c,
            norm=norm_c,
            interpolation="nearest",
        )
    else:
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

    if classes is not None:
        cax = fig.add_axes([0.25, 0.05, 0.5, 0.03])
        classes.colorbar(fig, im, cax, label=colorbar_label)
    else:
        cbar = fig.colorbar(im, extend=colorbar_extend)
        cbar.set_label(colorbar_label, labelpad=2)

    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.grid(True, ls="--", alpha=0.3)

    def init():
        im.set_data(np.ma.masked_all(all_data[0].shape))
        title.set_text("")
        return [im, title]

    def update(frame):
        im.set_data(all_data[frame])
        title.set_text(f"{title_prefix} – Año {years[frame]}")
        return [im, title]

    anim = FuncAnimation(
        fig, update, frames=len(all_data), init_func=init, blit=True, interval=interval
    )
    return fig, ax, anim


def save_all_lands_animation(
    anim,
    *,
    gif_path: str | os.PathLike[str] | None = None,
    mp4_path: str | os.PathLike[str] | None = None,
    fps: int = 3,
    dpi: int = 150,
):
    from matplotlib.animation import FFMpegWriter, PillowWriter

    if gif_path is not None:
        Path(os.fspath(gif_path)).parent.mkdir(parents=True, exist_ok=True)
        anim.save(os.fspath(gif_path), writer=PillowWriter(fps=fps))
    if mp4_path is not None:
        Path(os.fspath(mp4_path)).parent.mkdir(parents=True, exist_ok=True)
        writer = FFMpegWriter(fps=fps, bitrate=1800)
        anim.save(os.fspath(mp4_path), writer=writer, dpi=dpi)


def make_all_lands_animation(
    *,
    results_root: str | os.PathLike[str],
    variable: str,
    start_year: int,
    end_year: int,
    out_dir: str | os.PathLike[str] | None = None,
    gadm_peru_gpkg: str | os.PathLike[str] | None = DEFAULT_GADM_PERU_GPKG,
    title_prefix: str | None = None,
    vmax_percentile: float | None = 99.0,
    fps: int = 3,
    gif: bool = True,
    mp4: bool = True,
    classed: bool = True,
    n_classes: int = 6,
) -> dict[str, str]:
    """
    Build the yearly "all lands" animation for one variable (APP, BU or FU).

    Sums the three cultivo rasters of each year (each point belongs to one
    region, so the sum is that point's value) and writes
    ``<out_dir>/<stem>.gif`` / ``.mp4`` (``out_dir`` defaults to
    ``results_root``).  ``vmax_percentile`` caps the colour scale at that
    percentile of all frames so a few extreme pixels do not wash out the map
    (the colourbar then shows an arrow); ``None`` uses the true maximum.

    By default (``classed=True``) the colours are ``n_classes`` discrete
    classes of one hue (``viz_palette.sequential_classes``), the same for all
    frames; the eye reads discrete steps, not a continuous ramp.
    ``vmax_percentile`` then only applies with ``classed=False``.
    """
    spec = ANIMATION_VARIABLES[variable.upper()]
    results_root = Path(os.fspath(results_root))
    out_dir = Path(os.fspath(out_dir)) if out_dir is not None else results_root

    paths_by_year = collect_paths_by_year(
        results_root=results_root,
        start_year=start_year,
        end_year=end_year,
        raster_glob=spec["raster_glob"],
        strict=False,
    )
    years = sorted(paths_by_year)
    all_data, extent, ref_crs, vmin, vmax, _ = build_all_lands_data(paths_by_year)
    extend = "neither"
    classes = None
    if classed:
        from viz_palette import sequential_classes

        # same classes for every frame, so a colour means the same in every year
        classes = sequential_classes(np.ma.stack(all_data).compressed(), n_classes=n_classes)
    elif vmax_percentile is not None:
        vals = np.ma.stack(all_data).compressed()
        if vals.size:
            capped = float(np.percentile(vals, vmax_percentile))
            if capped < float(vmax):
                vmax, extend = capped, "max"

    gadm0 = gadm1 = None
    if gadm_peru_gpkg is not None:
        try:
            gadm0, gadm1 = load_gadm_peru(gadm_peru_gpkg)
        except Exception as e:
            logger.warning("Could not load GADM boundaries: %s", e)

    import matplotlib.pyplot as plt

    fig, _, anim = create_all_lands_animation(
        all_data=all_data,
        extent=extent,
        years=years,
        gadm_admin0=gadm0,
        gadm_admin1=gadm1,
        ref_crs=ref_crs,
        vmin=vmin,
        vmax=vmax,
        colorbar_label=spec["label"],
        title_prefix=title_prefix or f"SIMCAST {variable.upper()}",
        colorbar_extend=extend,
        classes=classes,
    )
    out = {}
    gif_path = out_dir / f"{spec['stem']}.gif" if gif else None
    mp4_path = out_dir / f"{spec['stem']}.mp4" if mp4 else None
    save_all_lands_animation(anim, gif_path=gif_path, mp4_path=mp4_path, fps=fps)
    plt.close(fig)
    if gif_path:
        out["gif"] = os.fspath(gif_path)
    if mp4_path:
        out["mp4"] = os.fspath(mp4_path)
    logger.info("%s animation: %d years (%d..%d), vmax=%.1f -> %s",
                variable, len(years), years[0], years[-1], float(vmax), out)
    return out
