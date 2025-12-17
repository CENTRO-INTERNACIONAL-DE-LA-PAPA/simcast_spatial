import glob
import os
from pathlib import Path

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
from dask.distributed import Client, as_completed
from tqdm.auto import tqdm

DEFAULT_GADM_PERU_GPKG = "https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_PER.gpkg"


def find_simcast_result_files(
    results_root: str | os.PathLike[str],
    *,
    parquet_name: str = "simcast_results.parquet",
) -> list[str]:
    """Find all SIMCAST result parquet files under the standard `<year>/<period>/<region>/` layout."""
    results_root = os.fspath(results_root)
    pattern = os.path.join(results_root, "*", "*", "*", parquet_name)
    return sorted(glob.glob(pattern))


def validate_files(
    results_root: str | os.PathLike[str],
    *,
    parquet_name: str = "simcast_results.parquet",
    engine: str = "pyarrow",
) -> list[str]:
    """
    Read each parquet file to identify corrupted ones.

    Returns a list of corrupted file paths (empty list means all readable).
    """
    files = find_simcast_result_files(results_root, parquet_name=parquet_name)
    corrupted = []
    for f in tqdm(files, desc="Validating Files"):
        try:
            pd.read_parquet(f, engine=engine, columns=["ID"])
        except Exception:
            corrupted.append(f)
    return corrupted


def run_ols_for_batch(
    id_batch: list,
    file_list: list,
    year_range: tuple,
    *,
    min_years: int = 10,
) -> tuple[list[dict], pd.DataFrame]:
    """
    Performs OLS regression for APP, BU, and FU yearly aggregates for a batch of grid IDs.

    - APP: yearly max(APP)
    - ABU: yearly sum(BU)
    - AFU: yearly sum(FU)

    Returns:
      - stats_results: list of dicts (one per ID)
      - yearly_data: DataFrame with columns [ID, year, APP, ABU, AFU]
    """
    # Local import so plotting helpers can be used without statsmodels installed.
    import statsmodels.api as sm

    stats_results: list[dict] = []
    yearly_data_export: list[pd.DataFrame] = []

    def _fit_ols(data: pd.DataFrame, y_col: str, x_col: str = "year") -> dict:
        nan_res = {
            "slope": np.nan,
            "p_value": np.nan,
            "std_err": np.nan,
            "conf_int_lower": np.nan,
            "conf_int_upper": np.nan,
            "r_squared": np.nan,
        }

        clean_data = data.dropna(subset=[y_col])
        if len(clean_data) < min_years:
            return {f"{k}_{y_col.lower()}": v for k, v in nan_res.items()}

        X = sm.add_constant(clean_data[x_col])
        y = clean_data[y_col]
        model = sm.OLS(y, X).fit()
        conf_int = model.conf_int().loc[x_col]

        results_dict = {
            "slope": model.params[x_col],
            "p_value": model.pvalues[x_col],
            "std_err": model.bse[x_col],
            "conf_int_lower": conf_int[0],
            "conf_int_upper": conf_int[1],
            "r_squared": model.rsquared,
        }
        return {f"{k}_{y_col.lower()}": v for k, v in results_dict.items()}

    try:
        df_batch = pd.read_parquet(file_list, filters=[("ID", "in", id_batch)])
        if df_batch.empty:
            return [], pd.DataFrame()

        df_batch["year"] = pd.to_datetime(df_batch["date"]).dt.year

        for id_val, df_id in df_batch.groupby("ID"):
            try:
                yearly_data = (
                    df_id.groupby("year")
                    .agg(APP=("APP", "max"), ABU=("BU", "sum"), AFU=("FU", "sum"))
                    .reset_index()
                )

                yearly_data_filtered = yearly_data[
                    (yearly_data["year"] > year_range[0])
                    & (yearly_data["year"] < year_range[1])
                ].copy()

                app_results = _fit_ols(yearly_data_filtered, "APP")
                abu_results = _fit_ols(yearly_data_filtered, "ABU")
                afu_results = _fit_ols(yearly_data_filtered, "AFU")

                stats_results.append(
                    {"ID": id_val, **app_results, **abu_results, **afu_results}
                )

                yearly_data_to_export = yearly_data[
                    ["year", "APP", "ABU", "AFU"]
                ].copy()
                yearly_data_to_export["ID"] = id_val
                yearly_data_export.append(yearly_data_to_export)
            except Exception:
                continue

        yearly_df = (
            pd.concat(yearly_data_export) if yearly_data_export else pd.DataFrame()
        )
        return stats_results, yearly_df

    except Exception:
        return [], pd.DataFrame()


def compute_trend_stats(
    *,
    results_root: str | os.PathLike[str],
    potato_grid_file: str | os.PathLike[str],
    output_stats_parquet_path: str | os.PathLike[str],
    output_yearly_data_path: str | os.PathLike[str] | None = None,
    year_range: tuple = (1981, 2020),
    batch_size: int = 500,
    min_years: int = 10,
    n_workers: int = 14,
    threads_per_worker: int = 2,
    memory_limit: str | None = "16GB",
) -> pd.DataFrame:
    """
    Compute OLS trend stats for APP/ABU/AFU across years for each grid point.

    Reads all `simcast_results.parquet` files under `results_root`, aggregates yearly metrics
    per ID, fits OLS(year -> metric), and writes a stats parquet.

    Uses Dask Distributed to parallelize across ID batches.
    """
    results_root = os.fspath(results_root)

    files = find_simcast_result_files(results_root)
    if not files:
        raise FileNotFoundError(
            f"No simcast_results.parquet files found under {results_root}"
        )

    potato_grid = gpd.read_file(os.fspath(potato_grid_file))
    if "ID" not in potato_grid.columns:
        potato_grid["ID"] = potato_grid.index
    unique_ids = potato_grid["ID"].unique().tolist()

    client = Client(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
    )
    print(f"Dask dashboard: {client.dashboard_link}")

    # Scatter file list once to reduce graph/serialization overhead.
    files_future = client.scatter(files, broadcast=True)

    id_batches = [
        unique_ids[i : i + batch_size] for i in range(0, len(unique_ids), batch_size)
    ]
    tasks = [
        dask.delayed(run_ols_for_batch)(
            batch, files_future, year_range, min_years=min_years
        )
        for batch in id_batches
    ]
    futures = client.compute(tasks)

    stats_results_list: list[dict] = []
    yearly_data_list: list[pd.DataFrame] = []
    for _, (stats_batch, yearly_data_batch) in tqdm(
        as_completed(futures, with_results=True),
        total=len(futures),
        desc="Processing Batches",
    ):
        if stats_batch:
            stats_results_list.extend(stats_batch)
        if isinstance(yearly_data_batch, pd.DataFrame) and not yearly_data_batch.empty:
            yearly_data_list.append(yearly_data_batch)

    client.close()

    stats_df = pd.DataFrame(stats_results_list)
    Path(output_stats_parquet_path).parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_parquet(output_stats_parquet_path, index=False)

    if output_yearly_data_path is not None and yearly_data_list:
        yearly_df = pd.concat(yearly_data_list)
        Path(output_yearly_data_path).parent.mkdir(parents=True, exist_ok=True)
        yearly_df.to_parquet(output_yearly_data_path, index=False)

    return stats_df


def create_significance_map(
    *,
    shapefile_path: str | os.PathLike[str],
    stats_parquet_path: str | os.PathLike[str],
    output_image_path: str | os.PathLike[str],
    variable_prefix: str,
    p_threshold: float = 0.05,
    gadm_peru_gpkg: str | os.PathLike[str] | None = DEFAULT_GADM_PERU_GPKG,
    markersize: float = 2,
    figsize: tuple[float, float] = (12, 12),
):
    """
    Plot a significance map (increase/decrease/not significant) for a variable prefix.

    `variable_prefix` must match the stats parquet columns, e.g.:
      - "app" -> p_value_app + slope_app
      - "abu" -> p_value_abu + slope_abu
      - "afu" -> p_value_afu + slope_afu
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    p_value_col = f"p_value_{variable_prefix}"
    slope_col = f"slope_{variable_prefix}"

    potato_grid = gpd.read_file(os.fspath(shapefile_path))
    if "ID" not in potato_grid.columns:
        potato_grid["ID"] = potato_grid.index

    stats_df = pd.read_parquet(stats_parquet_path)
    if p_value_col not in stats_df.columns or slope_col not in stats_df.columns:
        raise ValueError(
            f"Required columns '{p_value_col}' and '{slope_col}' not found in {stats_parquet_path}"
        )

    trend_gdf = potato_grid.merge(stats_df, on="ID", how="left")

    # 0=NoData, 1=NotSignificant, 2=SignificantIncrease, 3=SignificantDecrease
    trend_gdf["trend_type"] = 0
    valid_mask = trend_gdf[p_value_col].notna()
    trend_gdf.loc[
        valid_mask & (trend_gdf[p_value_col] >= p_threshold), "trend_type"
    ] = 1
    significant_mask = valid_mask & (trend_gdf[p_value_col] < p_threshold)
    trend_gdf.loc[significant_mask & (trend_gdf[slope_col] > 0), "trend_type"] = 2
    trend_gdf.loc[significant_mask & (trend_gdf[slope_col] < 0), "trend_type"] = 3

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    colors = {
        2: "#D94535",  # Increase
        3: "#009B77",  # Decrease
        1: "#E6E7E8",  # Not significant
        0: "#FFFFFF00",  # Transparent
    }
    trend_gdf["color"] = trend_gdf["trend_type"].map(colors)
    trend_gdf.plot(color=trend_gdf["color"], ax=ax, markersize=markersize)

    if gadm_peru_gpkg is not None:
        try:
            gadm_admin0 = gpd.read_file(gadm_peru_gpkg, layer="ADM_ADM_0").to_crs(
                potato_grid.crs
            )
            gadm_admin0.boundary.plot(
                ax=ax, edgecolor="black", linewidth=1.2, zorder=10
            )
        except Exception as e:
            print(f"Could not load GADM boundaries: {e}")

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Significant Increase",
            markerfacecolor=colors[2],
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Significant Decrease",
            markerfacecolor=colors[3],
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="No Significant Trend",
            markerfacecolor=colors[1],
            markersize=10,
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        title=f"{variable_prefix.upper()} Trend (p < {p_threshold})",
        fontsize=12,
    )

    ax.set_title(
        f"Significance of {variable_prefix.upper()} Trends", fontsize=16, pad=20
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    Path(output_image_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_image_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_coefficient_map(
    *,
    shapefile_path: str | os.PathLike[str],
    stats_parquet_path: str | os.PathLike[str],
    output_image_path: str | os.PathLike[str],
    variable_prefix: str,
    p_threshold: float = 0.05,
    gadm_peru_gpkg: str | os.PathLike[str] | None = DEFAULT_GADM_PERU_GPKG,
    markersize: float = 2,
    figsize: tuple[float, float] = (12, 12),
):
    """
    Plot a coefficient (slope) map for significant points for a variable prefix.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    p_value_col = f"p_value_{variable_prefix}"
    slope_col = f"slope_{variable_prefix}"

    potato_grid = gpd.read_file(os.fspath(shapefile_path))
    if "ID" not in potato_grid.columns:
        potato_grid["ID"] = potato_grid.index

    stats_df = pd.read_parquet(stats_parquet_path)
    if p_value_col not in stats_df.columns or slope_col not in stats_df.columns:
        raise ValueError(
            f"Required columns '{p_value_col}' and '{slope_col}' not found in {stats_parquet_path}"
        )

    trend_gdf = potato_grid.merge(stats_df, on="ID", how="left")

    significant_mask = trend_gdf[p_value_col].notna() & (
        trend_gdf[p_value_col] < p_threshold
    )
    significant_gdf = trend_gdf[significant_mask].copy()
    nonsignificant_gdf = trend_gdf[~significant_mask].copy()

    pantone_green = "#009B77"
    pantone_red = "#D94535"
    neutral_color = "#F5F5F5"
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(
        "GreenRedDiverging", [pantone_green, neutral_color, pantone_red]
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    nonsignificant_gdf.plot(ax=ax, color="#E6E7E8", markersize=markersize)

    if not significant_gdf.empty:
        max_abs_slope = significant_gdf[slope_col].abs().max()
        vmin, vmax = -max_abs_slope, max_abs_slope
        significant_gdf.plot(
            ax=ax,
            column=slope_col,
            cmap=custom_cmap,
            markersize=markersize,
            vmin=vmin,
            vmax=vmax,
            legend=True,
            legend_kwds={
                "label": f"{variable_prefix.upper()} Trend Coefficient",
                "orientation": "horizontal",
                "pad": 0.01,
            },
        )
    else:
        print(
            f"No significant points found for {variable_prefix.upper()} at p < {p_threshold}"
        )

    if gadm_peru_gpkg is not None:
        try:
            gadm_admin0 = gpd.read_file(gadm_peru_gpkg, layer="ADM_ADM_0").to_crs(
                potato_grid.crs
            )
            gadm_admin0.boundary.plot(
                ax=ax, edgecolor="black", linewidth=1.2, zorder=10
            )
        except Exception as e:
            print(f"Could not load GADM boundaries: {e}")

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="No Significant Trend",
            markerfacecolor="#E6E7E8",
            markersize=10,
        )
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    ax.set_title(
        f"Magnitude of {variable_prefix.upper()} Trends (p < {p_threshold})",
        fontsize=16,
        pad=20,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    Path(output_image_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_image_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
