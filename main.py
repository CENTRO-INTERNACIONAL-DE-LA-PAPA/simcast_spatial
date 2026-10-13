"""
Pipeline runner.

Examples
--------
Build inputs (PISCOp v3.0 prec + reused tmax/tmin/td)::

    uv run python main.py --steps prepare

Full rerun for resistant and susceptible varieties::

    uv run python main.py --steps simcast rasters merge trend maps animations --vt r s

Steps run in the order listed in STEP_ORDER, whatever order they are given in.
Paths live in `config.py` (`EXAMPLE_SOURCES`, `EXAMPLE_PATHS`, `RESULTS_ROOTS`).
"""

from __future__ import annotations

import argparse
import logging
import time

STEP_ORDER = [
    "prepare",  # build <parquet_base_dir> (prepare_inputs.py)
    "resampling",  # legacy: prec NetCDF -> 0.01 deg NetCDFs (resampling.py)
    "masking",  # legacy: NetCDF -> parquet (masking.py)
    "simcast",  # SIMCAST per season -> simcast_results.parquet + APP tif
    "rasters",  # BU/FU season-sum GeoTIFFs
    "merge",  # one GeoTIFF per year and variable (all regions) for ArcGIS
    "points",  # add lon/lat/altitude to existing simcast_results.parquet files
    "trend",  # OLS trend stats + yearly table (with lon/lat/altitude)
    "maps",  # trend significance / coefficient PNGs
    "animations",  # APP, BU, FU "all lands" GIF + MP4
    "elnino",  # APP in El Niño years vs. non-Niño average, KL divergence + figure
]

VT_NAMES = {"s": "susceptible", "ms": "moderately susceptible", "mr": "moderately resistant",
            "r": "resistant", "hr": "highly resistant"}


def run(steps, varieties, *, start_year=1981, end_year=2020, animation_vars=("APP", "BU", "FU")):
    from dataclasses import replace

    from config import (
        EXAMPLE_PATHS,
        EXAMPLE_SOURCES,
        RESULTS_ROOTS,
        ResourcesConfig,
        SimcastConfig,
    )

    paths = EXAMPLE_PATHS
    resources = ResourcesConfig()
    steps = [s for s in STEP_ORDER if s in set(steps)]

    # ------------------------------------------------------------------
    # Variety-independent steps
    # ------------------------------------------------------------------
    if "prepare" in steps:
        from prepare_inputs import prepare_all

        prepare_all(EXAMPLE_SOURCES, workers=resources.prepare_workers)

    if "resampling" in steps:
        from resampling import run_resampling

        run_resampling(
            data_file=paths.prec_data_file,
            grid_file=paths.resample_grid_file,
            output_dir=paths.prec_resample_output_dir,
            max_workers=resources.resampling_workers,
        )

    if "masking" in steps:
        from masking import extract_folder_to_parquets

        masking_jobs = [
            # variable, netcdf_dir, glob_pattern, var_name_in_netcdf
            ("prec", paths.prec_resample_output_dir, "prec_daily_*.nc", "Prec"),
            ("tmax", paths.tmax_nc_dir, "tmax_daily_*.nc", "tmax"),
            ("tmin", paths.tmin_nc_dir, "tmin_daily_*.nc", "tmin"),
            ("td", paths.td_nc_dir, "td_daily_*.nc", "td"),
        ]
        for variable, netcdf_dir, glob_pattern, nc_var_name in masking_jobs:
            extract_folder_to_parquets(
                resample_folder=netcdf_dir,
                output_folder=paths.parquet_base_dir / variable / "Outputs",
                potato_grid_file=paths.potato_grid_file,
                glob_pattern=glob_pattern,
                var_name=nc_var_name,
                time_name="time",
                chunk_size=resources.masking_chunk_size,
                max_workers=resources.masking_workers,
            )

    # ------------------------------------------------------------------
    # Per-variety steps
    # ------------------------------------------------------------------
    for vt in varieties:
        sim_cfg = SimcastConfig(vt=vt)
        results_root = RESULTS_ROOTS[vt]
        vpaths = replace(paths, results_root=results_root)
        label = VT_NAMES.get(vt, vt)
        print(f"\n===== variety {vt!r} ({label}) -> {results_root} =====", flush=True)
        t0 = time.perf_counter()

        if "simcast" in steps:
            from load_variables import parallel_simcast_pipeline

            parallel_simcast_pipeline(
                start_year=start_year,
                end_year=end_year,
                vt=sim_cfg.vt,
                base_path=vpaths.parquet_base_dir,
                potato_grid_file=vpaths.potato_grid_file,
                results_root=results_root,
                n_workers=resources.dask_n_workers,
                threads_per_worker=resources.dask_threads_per_worker,
                memory_limit=resources.dask_memory_limit,
                dashboard_address=resources.dask_dashboard_address,
                write_tiff=True,  # APP raster per season (used by the APP animation)
                show_task_bar=False,
                id_chunk_size=resources.simcast_id_chunk_size,
                skip_existing=True,  # rerun resumes where it stopped
                max_data_year=end_year,
                rh_thresh=sim_cfg.rh_thresh,
                timezone=sim_cfg.timezone,
                min_day=sim_cfg.min_day,
                forced_day=sim_cfg.forced_day,
                sun_method=sim_cfg.sun_method,
            )
            print(f"[{vt}] simcast done ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)

        if "rasters" in steps:
            from animation_all_lands import generate_bu_fu_rasters

            generate_bu_fu_rasters(
                results_root=results_root,
                grid_file_path=vpaths.potato_grid_file,
                workers=resources.prepare_workers,
                skip_existing=True,
            )

        if "merge" in steps:
            from merge_rasters import merge_all_years

            merge_all_years(
                results_root,
                variety=VT_NAMES.get(vt, vt).replace(" ", "_"),
                overwrite=True,  # cheap (~10 s); keeps merged files in sync
            )

        if "points" in steps:
            from point_attrs import backfill_results

            backfill_results(results_root, vpaths.potato_grid_file, workers=6)

        stats_parquet_path = results_root / "statistical_trends_all.parquet"
        yearly_parquet_path = results_root / "yearly_aggregated_data.parquet"

        if "trend" in steps:
            from trend_significance import compute_trend_stats

            compute_trend_stats(
                results_root=results_root,
                potato_grid_file=vpaths.potato_grid_file,
                output_stats_parquet_path=stats_parquet_path,
                output_yearly_data_path=yearly_parquet_path,
                year_range=(start_year, end_year),  # inclusive
                min_years=10,
                aggregate_by="season",  # "calendar_year" = legacy behaviour
                exclude_truncated=True,  # drop seasons cut short by missing inputs
                workers=resources.prepare_workers,
            )

        if "maps" in steps:
            from trend_significance import create_coefficient_map, create_significance_map

            for var in ["app", "abu", "afu"]:
                create_significance_map(
                    shapefile_path=vpaths.potato_grid_file,
                    stats_parquet_path=stats_parquet_path,
                    output_image_path=results_root / f"trend_significance_map_{var}.png",
                    variable_prefix=var,
                    p_threshold=0.05,
                    gadm_peru_gpkg=vpaths.gadm_peru_gpkg,
                )
                create_coefficient_map(
                    shapefile_path=vpaths.potato_grid_file,
                    stats_parquet_path=stats_parquet_path,
                    output_image_path=results_root / f"coefficient_magnitude_map_{var}.png",
                    variable_prefix=var,
                    p_threshold=0.05,
                    gadm_peru_gpkg=vpaths.gadm_peru_gpkg,
                )

        if "animations" in steps:
            from animation_all_lands import make_all_lands_animation

            for var in animation_vars:
                make_all_lands_animation(
                    results_root=results_root,
                    variable=var,
                    start_year=start_year,
                    end_year=end_year,
                    gadm_peru_gpkg=vpaths.gadm_peru_gpkg,
                    title_prefix=f"SIMCAST {var} ({label})",
                    vmax_percentile=99.0,
                    fps=3,
                )

        if "elnino" in steps:
            from pathlib import Path

            from el_nino_analysis import run_el_nino_analysis

            run_el_nino_analysis(
                results_root=results_root,
                variety=label.replace(" ", "_"),
                variety_label={"resistant": "resistente", "susceptible": "susceptible"}.get(label, label),
                events_csv=Path(__file__).parent / "reference" / "el_nino_costero_enfen2024.csv",
                years=range(start_year, min(end_year, 2019) + 1),  # 2020 lacks Sierra Alta
                min_overlap_months=3,
                min_magnitude="moderado",
                gadm_peru_gpkg=vpaths.gadm_peru_gpkg,
            )

        print(f"[{vt}] finished in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", choices=STEP_ORDER, required=True)
    ap.add_argument("--vt", nargs="+", default=["r"], choices=list(VT_NAMES),
                    help="variety resistance classes to run (default: r)")
    ap.add_argument("--start-year", type=int, default=1981)
    ap.add_argument("--end-year", type=int, default=2020)
    ap.add_argument("--animation-vars", nargs="+", default=["APP", "BU", "FU"])
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(a.steps, a.vt, start_year=a.start_year, end_year=a.end_year,
        animation_vars=tuple(a.animation_vars))


if __name__ == "__main__":
    main()
