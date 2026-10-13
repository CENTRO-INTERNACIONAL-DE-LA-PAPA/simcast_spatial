"""
Merge the per-region season rasters of each year into one GeoTIFF per
variable, at ``<results_root>/<year>/``, ready to open in ArcGIS / QGIS.

Input  (written by the SIMCAST and raster steps)::

    <results_root>/<year>/<cultivo>/<region>/simcast_<year>_<cultivo>_<region>.tif   APP (max)
    <results_root>/<year>/<cultivo>/<region>/BU_sum_<year>_<cultivo>_<region>.tif    BU  (sum)
    <results_root>/<year>/<cultivo>/<region>/FU_sum_<year>_<cultivo>_<region>.tif    FU  (sum)

Output::

    <results_root>/<year>/APP_<variety>_<year>.tif
    <results_root>/<year>/BU_<variety>_<year>.tif
    <results_root>/<year>/FU_<variety>_<year>.tif

Every potato point belongs to exactly one region, so the regional rasters do
not overlap (checked: overlapping valid pixels raise an error) and the merged
value of a pixel is the value of its region's season.  The outputs are
single-band GeoTIFFs in EPSG:4326 with the same 0.01 deg grid as the inputs,
NoData = -9999, LZW compression, 256x256 tiles and internal overviews, plus
band description and metadata tags (variable, units, year, variety, seasons
and regions merged).  File names contain no spaces.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

NODATA = -9999

# variable -> (input file prefix, description, units)
VARIABLES = {
    "APP": ("simcast_", "Fungicide applications per season (max APP)", "applications"),
    "BU": ("BU_sum_", "Blight units, season sum", "blight units"),
    "FU": ("FU_sum_", "Fungicide units, season sum", "fungicide units"),
}

SEASON_MONTHS = {
    "cultivo1": "Nov(year)-Apr(year+1)",
    "cultivo2": "Jul-Dec",
    "cultivo3": "Apr-Jul",
}

_YEAR_RE = re.compile(r"^\d{4}$")


def _region_rasters(year_dir: Path, prefix: str) -> list[Path]:
    return sorted(Path(p) for p in glob.glob(os.fspath(year_dir / "*" / "*" / f"{prefix}*.tif")))


def merge_year(
    year_dir: str | os.PathLike[str],
    variable: str,
    *,
    variety: str,
    overwrite: bool = False,
    overviews: bool = True,
) -> Path | None:
    """Merge one variable for one year; returns the output path (None if no inputs)."""
    year_dir = Path(os.fspath(year_dir))
    year = year_dir.name
    prefix, description, units = VARIABLES[variable]
    out = year_dir / f"{variable}_{variety}_{year}.tif"
    if out.exists() and not overwrite:
        return out

    inputs = _region_rasters(year_dir, prefix)
    if not inputs:
        return None

    merged = None
    covered = None
    parts = []
    for p in inputs:
        cultivo, region = p.parent.parent.name, p.parent.name
        with rasterio.open(p) as src:
            data = src.read(1)
            nodata = src.nodata if src.nodata is not None else NODATA
            if merged is None:
                profile = src.profile.copy()
                ref = (src.shape, src.transform, src.crs)
                merged = np.full(src.shape, NODATA, dtype=data.dtype)
                covered = np.zeros(src.shape, dtype=bool)
            elif (src.shape, src.transform, src.crs) != ref:
                raise ValueError(f"Grid mismatch in {p}")
            valid = data != nodata
            overlap = int((valid & covered).sum())
            if overlap:
                raise ValueError(f"{overlap} pixels of {p} overlap another region")
            merged[valid] = data[valid]
            covered |= valid
            parts.append(f"{cultivo}:{region} ({SEASON_MONTHS.get(cultivo, '?')}, {int(valid.sum())} px)")

    profile.update(
        driver="GTiff",
        count=1,
        dtype=merged.dtype,
        nodata=NODATA,
        compress="lzw",
        predictor=3 if np.issubdtype(merged.dtype, np.floating) else 2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    tmp = out.with_suffix(".tmp.tif")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(merged, 1)
        dst.set_band_description(1, f"{variable} - {description}")
        dst.update_tags(
            variable=variable,
            description=description,
            units=units,
            year=year,
            variety=variety,
            season_year_note="cultivo1 runs Nov of <year> to Apr of <year+1>",
            regions_merged="; ".join(parts),
            n_valid_pixels=str(int(covered.sum())),
            source="SIMCAST-spatial (CIP); AI-assisted processing, review before use",
        )
        if overviews:
            dst.build_overviews([2, 4, 8], Resampling.nearest)
            dst.update_tags(ns="rio_overview", resampling="nearest")
    os.replace(tmp, out)
    return out


def merge_all_years(
    results_root: str | os.PathLike[str],
    *,
    variety: str,
    variables=("APP", "BU", "FU"),
    overwrite: bool = False,
) -> list[Path]:
    """Merge every ``<results_root>/<year>/`` folder for the given variables."""
    root = Path(os.fspath(results_root))
    written = []
    for year_dir in sorted(p for p in root.iterdir() if p.is_dir() and _YEAR_RE.match(p.name)):
        for var in variables:
            out = merge_year(year_dir, var, variety=variety, overwrite=overwrite)
            if out is not None:
                written.append(out)
    print(f"[merge] {root}: {len(written)} merged rasters ({variety})")
    return written
