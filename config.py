from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from prepare_inputs import InputSources


@dataclass(frozen=True)
class PathsConfig:
    """
    Filesystem locations used by the pipeline.

    These are intentionally explicit (no magic defaults) so other users can
    point to their own data layout without editing library code.
    """

    # Resampling inputs/outputs
    prec_data_file: Path
    resample_grid_file: Path
    prec_resample_output_dir: Path

    # NetCDF inputs for masking
    tmax_nc_dir: Path
    tmin_nc_dir: Path
    td_nc_dir: Path

    # Masking / spatial grid
    potato_grid_file: Path

    # Parquet layout expected by `load_variables.py`
    parquet_base_dir: Path

    # SIMCAST outputs (parquets + rasters)
    results_root: Path

    # Optional boundary overlay for animations (URL or local .gpkg)
    gadm_peru_gpkg: str | Path


@dataclass(frozen=True)
class ResourcesConfig:
    """
    Tunable CPU/RAM settings.

    Use smaller values on laptops / low-RAM machines.
    """

    # ProcessPoolExecutor limits
    resampling_workers: int | None = None
    masking_workers: int | None = None
    prepare_workers: int = 8

    # Masking extraction
    masking_chunk_size: int = 10_000

    # SIMCAST compute: IDs per Dask task. Each task loads the 4 climate
    # variables for its IDs only (~12 MB per variable per 1 000 IDs-season).
    # 5 000 gives ~60 tasks per year, enough to keep 20 workers balanced.
    simcast_id_chunk_size: int = 5_000

    # Dask distributed. The per-point work is pandas/Python-heavy, so use one
    # thread per worker process (threads would fight over the GIL): 20 single
    # threaded workers finished one year in ~66 s on the 32-core workstation.
    dask_n_workers: int = 20
    dask_threads_per_worker: int = 1
    dask_memory_limit: str | None = "5GB"
    dask_dashboard_address: str = ":8787"


@dataclass(frozen=True)
class SimcastConfig:
    """Model parameters that affect SIMCAST outputs."""

    vt: str = "r"
    rh_thresh: float = 90.0
    timezone: str = "America/Lima"
    min_day: int = 5
    forced_day: int = 25
    # "noaa": vectorised sunrise/sunset with a fixed UTC offset (fast, no DST
    # artefacts).  "astral": the original per-day astral calls (slow; inherits
    # historical DST periods of the tz database).
    sun_method: str = "noaa"


_DATA1 = Path("/media/ppalacios/Data1/henry_simcast_peru")

# Canonical SIMCAST inputs built by `prepare_inputs.prepare_all` (2026 rerun).
SIMCAST_INPUTS_DIR = _DATA1 / "simcast_inputs_v2"

EXAMPLE_SOURCES = InputSources(
    potato_grid_file=_DATA1 / "PotatoZonning/CENAGRO_OnlyPotatoes_Pisco_Altitude.shp",
    output_base=SIMCAST_INPUTS_DIR,
    # PISCOt v1.2 (1981-2020), one NetCDF per year
    tmax_nc_dir=_DATA1 / "_raw/v1_2/tmax",
    tmin_nc_dir=_DATA1 / "_raw/v1_2/tmin",
    # PISCOeo_pm dew point (1981-2016), one NetCDF per month
    td_nc_dir=_DATA1 / "_raw/v1_1/td",
    td_nc_years=(1981, 2016),
    # Dew point 2017-2020 predicted with TimesFM (local machine A019866)
    td_pred_dir=Path("/home/cvelasquez/tdew_timesfm/preds_real"),
    td_pred_start="2017-01-01",
    td_pred_n_buckets=8192,
    td_pred_column="pred",
    td_pred_years=(2017, 2020),
    # Precipitation parquets already extracted at the potato points (1981-2019)
    prec_parquet_dir=_DATA1 / "prec/Outputs",
    years=(1981, 2020),
    grid_index_file=_DATA1 / "tdew_medallion/silver/grid_index.parquet",
)

EXAMPLE_PATHS = PathsConfig(
    prec_data_file=_DATA1 / "prec/data.nc",
    resample_grid_file=_DATA1 / "tmin/tmin_daily_1981_01.nc",
    prec_resample_output_dir=_DATA1 / "prec/Resample_v1",
    tmax_nc_dir=_DATA1 / "tmax",
    tmin_nc_dir=_DATA1 / "tmin_v1",
    td_nc_dir=_DATA1 / "td",
    potato_grid_file=EXAMPLE_SOURCES.potato_grid_file,
    parquet_base_dir=SIMCAST_INPUTS_DIR,
    results_root=_DATA1 / "results_resistant_v2",
    gadm_peru_gpkg="https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_PER.gpkg",
)
