def main():
    """
    Example "runner" script.

    Edit the paths/toggles below to run individual steps.
    """

    from config import EXAMPLE_PATHS, EXAMPLE_SOURCES, ResourcesConfig, SimcastConfig

    # ------------------------------------------------------------------
    # 0) EDIT ME: paths + compute settings
    # ------------------------------------------------------------------
    paths = EXAMPLE_PATHS
    sources = EXAMPLE_SOURCES
    resources = ResourcesConfig(
        resampling_workers=6,
        masking_workers=6,
        prepare_workers=8,
        masking_chunk_size=10_000,
        simcast_id_chunk_size=5_000,
        dask_n_workers=20,
        dask_threads_per_worker=1,
        dask_memory_limit="5GB",
    )
    sim_cfg = SimcastConfig(vt="r", timezone="America/Lima", sun_method="noaa")

    START_YEAR = 1981
    END_YEAR = 2020  # last year with tmax/tmin/td inputs (prec ends 2019-12)

    # ------------------------------------------------------------------
    # 1) Toggle steps on/off
    # ------------------------------------------------------------------
    RUN_PREPARE_INPUTS = False  # build <parquet_base_dir> from the 2026 sources
    RUN_RESAMPLING = False  # legacy: prec NetCDF resampling
    RUN_MASKING = False  # legacy: NetCDF -> parquet with masking.py
    RUN_SIMCAST = False
    RUN_BU_FU_RASTERS = False
    RUN_ALL_LANDS_ANIMATION = False
    RUN_TREND_STATS = False
    RUN_TREND_SIGNIFICANCE_MAPS = False
    RUN_TREND_COEFFICIENT_MAPS = False

    # ------------------------------------------------------------------
    # 2a) Input preparation (2026 rerun): PISCOt v1.2 tmax/tmin, PISCOeo_pm td
    #     (1981-2016), TimesFM td (2017-2020), existing prec parquets.
    # ------------------------------------------------------------------
    if RUN_PREPARE_INPUTS:
        from prepare_inputs import prepare_all

        prepare_all(sources, workers=resources.prepare_workers)

    # ------------------------------------------------------------------
    # 2b) Legacy resampling (prec only)
    # ------------------------------------------------------------------
    if RUN_RESAMPLING:
        from resampling import run_resampling

        run_resampling(
            data_file=paths.prec_data_file,
            grid_file=paths.resample_grid_file,
            output_dir=paths.prec_resample_output_dir,
            max_workers=resources.resampling_workers,
        )

    # ------------------------------------------------------------------
    # 3) Legacy masking / extraction (NetCDF -> monthly Parquet)
    # ------------------------------------------------------------------
    if RUN_MASKING:
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
    # 4) SIMCAST (Parquet -> results)
    # ------------------------------------------------------------------
    if RUN_SIMCAST:
        from load_variables import parallel_simcast_pipeline

        parallel_simcast_pipeline(
            start_year=START_YEAR,
            end_year=END_YEAR,
            vt=sim_cfg.vt,
            base_path=paths.parquet_base_dir,
            potato_grid_file=paths.potato_grid_file,
            results_root=paths.results_root,
            n_workers=resources.dask_n_workers,
            threads_per_worker=resources.dask_threads_per_worker,
            memory_limit=resources.dask_memory_limit,
            dashboard_address=resources.dask_dashboard_address,
            write_tiff=True,
            show_task_bar=True,
            id_chunk_size=resources.simcast_id_chunk_size,
            skip_existing=True,  # rerun resumes where it stopped
            max_data_year=END_YEAR,
            rh_thresh=sim_cfg.rh_thresh,
            timezone=sim_cfg.timezone,
            min_day=sim_cfg.min_day,
            forced_day=sim_cfg.forced_day,
            sun_method=sim_cfg.sun_method,
        )

    # ------------------------------------------------------------------
    # 5) Trend analysis (OLS over years)
    # ------------------------------------------------------------------
    stats_parquet_path = paths.results_root / "statistical_trends_all.parquet"
    yearly_parquet_path = paths.results_root / "yearly_aggregated_data.parquet"

    if RUN_TREND_STATS:
        from trend_significance import compute_trend_stats

        compute_trend_stats(
            results_root=paths.results_root,
            potato_grid_file=paths.potato_grid_file,
            output_stats_parquet_path=stats_parquet_path,
            output_yearly_data_path=yearly_parquet_path,
            year_range=(START_YEAR, END_YEAR),  # inclusive
            min_years=10,
            aggregate_by="season",  # "calendar_year" = legacy behaviour
            exclude_truncated=True,  # drop seasons cut short by missing inputs
            workers=resources.prepare_workers,
        )

    if RUN_TREND_SIGNIFICANCE_MAPS:
        from trend_significance import create_significance_map

        for var in ["app", "abu", "afu"]:
            create_significance_map(
                shapefile_path=paths.potato_grid_file,
                stats_parquet_path=stats_parquet_path,
                output_image_path=paths.results_root
                / f"trend_significance_map_{var}.png",
                variable_prefix=var,
                p_threshold=0.05,
                gadm_peru_gpkg=paths.gadm_peru_gpkg,
            )

    if RUN_TREND_COEFFICIENT_MAPS:
        from trend_significance import create_coefficient_map

        for var in ["app", "abu", "afu"]:
            create_coefficient_map(
                shapefile_path=paths.potato_grid_file,
                stats_parquet_path=stats_parquet_path,
                output_image_path=paths.results_root
                / f"coefficient_magnitude_map_{var}.png",
                variable_prefix=var,
                p_threshold=0.05,
                gadm_peru_gpkg=paths.gadm_peru_gpkg,
            )

    # ------------------------------------------------------------------
    # 6) BU/FU rasters from existing simcast_results.parquet
    # ------------------------------------------------------------------
    if RUN_BU_FU_RASTERS:
        from animation_all_lands import generate_bu_fu_rasters

        generate_bu_fu_rasters(
            results_root=paths.results_root,
            grid_file_path=paths.potato_grid_file,
            workers=resources.prepare_workers,
            skip_existing=True,
        )

    # ------------------------------------------------------------------
    # 7) "All lands" animation (sums BU rasters across cultivos)
    # ------------------------------------------------------------------
    if RUN_ALL_LANDS_ANIMATION:
        from animation_all_lands import (
            build_all_lands_data,
            collect_paths_by_year,
            create_all_lands_animation,
            load_gadm_peru,
            save_all_lands_animation,
        )

        paths_by_year = collect_paths_by_year(
            results_root=paths.results_root,
            start_year=START_YEAR,
            end_year=END_YEAR,
            strict=False,  # keep years even if one cultivo is missing
        )
        years = sorted(paths_by_year)

        all_data, extent, ref_crs, vmin, vmax, _ = build_all_lands_data(paths_by_year)

        gadm0, gadm1 = load_gadm_peru(paths.gadm_peru_gpkg)

        _, _, anim = create_all_lands_animation(
            all_data=all_data,
            extent=extent,
            years=years,
            gadm_admin0=gadm0,
            gadm_admin1=gadm1,
            ref_crs=ref_crs,
            vmin=vmin,
            vmax=vmax,
            interval=400,
            colorbar_label="Blight Units",
        )

        save_all_lands_animation(
            anim,
            gif_path=paths.results_root / "lateblight_all_lands_bu.gif",
            mp4_path=paths.results_root / "lateblight_all_lands_bu.mp4",
            fps=3,
        )


if __name__ == "__main__":
    main()
