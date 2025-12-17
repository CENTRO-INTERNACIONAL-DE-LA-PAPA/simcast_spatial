from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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

    # Masking extraction
    masking_chunk_size: int = 10_000

    # SIMCAST compute
    simcast_batch_size: int = (
        300  # IDs per in-memory batch inside each region-period task
    )

    # Dask distributed
    dask_n_workers: int = 8
    dask_threads_per_worker: int = 4
    dask_memory_limit: str | None = "14GB"
    dask_dashboard_address: str = ":8787"


@dataclass(frozen=True)
class SimcastConfig:
    """Model parameters that affect SIMCAST outputs."""

    vt: str = "r"
    rh_thresh: float = 90.0
    timezone: str = "America/Bogota"
    min_day: int = 5
    forced_day: int = 25


EXAMPLE_PATHS = PathsConfig(
    prec_data_file=Path("/media/ppalacios/Data1/henry_simcast_peru/prec/data.nc"),
    resample_grid_file=Path(
        "/media/ppalacios/Data1/henry_simcast_peru/tmin/tmin_daily_1981_01.nc"
    ),
    prec_resample_output_dir=Path(
        "/media/ppalacios/Data1/henry_simcast_peru/prec/Resample_v1"
    ),
    tmax_nc_dir=Path("/media/ppalacios/Data1/henry_simcast_peru/tmax"),
    tmin_nc_dir=Path("/media/ppalacios/Data1/henry_simcast_peru/tmin_v1"),
    td_nc_dir=Path("/media/ppalacios/Data1/henry_simcast_peru/td"),
    potato_grid_file=Path(
        "/media/ppalacios/Data1/henry_simcast_peru/"
        "PotatoZonning/CENAGRO_OnlyPotatoes_Pisco_Altitude.dbf"
    ),
    parquet_base_dir=Path("/media/ppalacios/Data/henry_simcast_peru"),
    results_root=Path("/media/ppalacios/Data/henry_simcast_peru/results_resistant"),
    gadm_peru_gpkg="https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_PER.gpkg",
)
