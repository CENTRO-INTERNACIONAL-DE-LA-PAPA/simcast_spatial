"""
Trend analysis of SIMCAST outputs: OLS of a yearly metric against the year,
per grid point.

Metrics (one row per (ID, season year)):

- ``APP``: final number of fungicide applications of the season (max APP)
- ``ABU``: accumulated blight units (sum BU)
- ``AFU``: accumulated fungicide units (sum FU)

Design (rewritten Oct 2026)
---------------------------
* Each season file ``<year>/<period>/<region>/simcast_results.parquet`` is read
  **once** and reduced to one row per ID.  The old implementation re-scanned
  every file for every batch of 500 IDs (~600 passes over ~20 GB).
* Metrics are aggregated **per season** (the ``<year>`` folder), not per
  calendar year.  ``cultivo1`` runs Nov-Apr, so a calendar-year aggregation
  mixed the end of one season with the start of the next and split the
  season's BU between two years.  ``aggregate_by="calendar_year"`` restores
  the legacy behaviour.
* Seasons flagged as truncated in ``coverage.json`` (e.g. cultivo1 2019,
  which ends in Dec 2019 because prec stops there) are excluded by default.
* The OLS fit (slope, t-test p-value, standard error, 95 % CI, R^2) is
  computed in closed form with NumPy for all IDs at once; it matches
  ``statsmodels.OLS`` to floating-point precision (``tests/test_trend.py``).
  ``year_range`` is inclusive at both ends.
"""

from __future__ import annotations

import concurrent.futures as cf
import glob
import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

DEFAULT_GADM_PERU_GPKG = "https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_PER.gpkg"

METRICS = ("APP", "ABU", "AFU")
STAT_NAMES = ("slope", "p_value", "std_err", "conf_int_lower", "conf_int_upper", "r_squared")


# ----------------------------------------------------------------------------
# Result files
# ----------------------------------------------------------------------------


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


def season_is_truncated(parquet_path: str | os.PathLike[str]) -> bool | None:
    """Read the `coverage.json` next to a season file; None when absent."""
    cov = Path(os.fspath(parquet_path)).parent / "coverage.json"
    if not cov.exists():
        return None
    with open(cov) as fh:
        return bool(json.load(fh).get("truncated", False))


# ----------------------------------------------------------------------------
# Per-season aggregation
# ----------------------------------------------------------------------------


def aggregate_season_file(
    parquet_path: str,
    *,
    aggregate_by: str = "season",
) -> pd.DataFrame:
    """
    Reduce one season file to [ID, year, period, region, APP, ABU, AFU, n_days].

    ``aggregate_by="season"`` uses the ``<year>`` folder as the year;
    ``"calendar_year"`` groups by the calendar year of each daily row
    (legacy behaviour).
    """
    p = Path(parquet_path)
    year_dir, period, region = p.parts[-4], p.parts[-3], p.parts[-2]
    df = pd.read_parquet(parquet_path, columns=["ID", "date", "APP", "BU", "FU"])
    if df.empty:
        return pd.DataFrame()

    if aggregate_by == "season":
        df["year"] = np.int32(int(year_dir))
    elif aggregate_by == "calendar_year":
        df["year"] = pd.to_datetime(df["date"]).dt.year.astype(np.int32)
    else:
        raise ValueError("aggregate_by must be 'season' or 'calendar_year'")

    agg = (
        df.groupby(["ID", "year"], sort=False)
        .agg(APP=("APP", "max"), ABU=("BU", "sum"), AFU=("FU", "sum"), n_days=("date", "size"))
        .reset_index()
    )
    agg["period"] = period
    agg["region"] = region
    agg["year"] = agg["year"].astype(np.int32)
    agg["n_days"] = agg["n_days"].astype(np.int32)
    return agg[["ID", "year", "period", "region", "APP", "ABU", "AFU", "n_days"]]


def build_yearly_table(
    results_root: str | os.PathLike[str],
    *,
    aggregate_by: str = "season",
    exclude_truncated: bool = True,
    workers: int = 8,
    files: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate every season file (in parallel) into one yearly table."""
    files = files or find_simcast_result_files(results_root)
    if not files:
        raise FileNotFoundError(f"No simcast_results.parquet files found under {results_root}")

    selected, dropped = [], []
    for f in files:
        if exclude_truncated and season_is_truncated(f):
            dropped.append(f)
        else:
            selected.append(f)
    if dropped:
        print(f"[trend] excluding {len(dropped)} truncated season(s):")
        for f in dropped:
            print("   ", Path(f).parent)

    parts = []
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(aggregate_season_file, f, aggregate_by=aggregate_by): f for f in selected}
        for fut in tqdm(cf.as_completed(futs), total=len(futs), desc="Aggregating seasons"):
            part = fut.result()
            if not part.empty:
                parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["ID", "year", "period", "region", *METRICS, "n_days"])

    yearly = pd.concat(parts, ignore_index=True)
    if aggregate_by == "calendar_year":
        # a calendar year can receive rows from two seasons of the same ID
        yearly = (
            yearly.groupby(["ID", "year"], sort=False)
            .agg(
                period=("period", "first"),
                region=("region", "first"),
                APP=("APP", "max"),
                ABU=("ABU", "sum"),
                AFU=("AFU", "sum"),
                n_days=("n_days", "sum"),
            )
            .reset_index()
        )
    return yearly.sort_values(["ID", "year"], kind="stable").reset_index(drop=True)


# ----------------------------------------------------------------------------
# Vectorised OLS
# ----------------------------------------------------------------------------


def ols_trend_by_id(
    yearly: pd.DataFrame,
    metric: str,
    *,
    min_years: int = 10,
    year_range: tuple[int, int] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    OLS ``metric ~ year`` for every ID, in closed form.

    Returns one row per ID with columns ``slope_<m>``, ``p_value_<m>``,
    ``std_err_<m>``, ``conf_int_lower_<m>``, ``conf_int_upper_<m>``,
    ``r_squared_<m>`` and ``n_years_<m>``; IDs with fewer than ``min_years``
    valid years get NaN statistics.  Identical to ``statsmodels.OLS`` with a
    constant (two-sided t-test, ``(1-alpha)`` CI), except that a metric that
    never changes over the years gets slope 0 / p-value 1 (statsmodels: NaN).
    """
    from scipy import stats as sps

    d = yearly[["ID", "year", metric]].dropna(subset=[metric])
    if year_range is not None:
        d = d[(d["year"] >= year_range[0]) & (d["year"] <= year_range[1])]

    ids = d["ID"].to_numpy()
    x = d["year"].to_numpy(dtype=np.float64)
    y = d[metric].to_numpy(dtype=np.float64)

    uniq, inv = np.unique(ids, return_inverse=True)
    n = np.bincount(inv).astype(np.float64)
    sx = np.bincount(inv, weights=x)
    sy = np.bincount(inv, weights=y)
    sxx = np.bincount(inv, weights=x * x)
    sxy = np.bincount(inv, weights=x * y)
    syy = np.bincount(inv, weights=y * y)

    with np.errstate(divide="ignore", invalid="ignore"):
        mx, my = sx / n, sy / n
        cxx = sxx - n * mx * mx  # sum (x - mx)^2
        cxy = sxy - n * mx * my
        cyy = syy - n * my * my
        slope = cxy / cxx
        sse = cyy - slope * cxy  # residual sum of squares
        sse = np.maximum(sse, 0.0)
        dof = n - 2
        sigma2 = sse / dof
        se = np.sqrt(sigma2 / cxx)
        t = slope / se
        p = 2.0 * sps.t.sf(np.abs(t), dof)
        tcrit = sps.t.ppf(1 - alpha / 2, dof)
        lo, hi = slope - tcrit * se, slope + tcrit * se
        r2 = np.where(cyy > 0, 1.0 - sse / cyy, np.nan)

    # A series that is constant over the years (e.g. ABU == 0 every season at
    # cold high-altitude points) has no trend: report slope 0 and p = 1 so the
    # maps show it as "not significant" instead of "no data".  (statsmodels
    # returns p = NaN here because the standard error is 0.)
    const = (cyy <= 0) & np.isfinite(cyy)
    slope[const], se[const], lo[const], hi[const] = 0.0, 0.0, 0.0, 0.0
    p[const] = 1.0

    bad = (n < max(min_years, 3)) | ~np.isfinite(cxx) | (cxx <= 0)
    for arr in (slope, p, se, lo, hi, r2):
        arr[bad] = np.nan

    m = metric.lower()
    return pd.DataFrame(
        {
            "ID": uniq,
            f"slope_{m}": slope,
            f"p_value_{m}": p,
            f"std_err_{m}": se,
            f"conf_int_lower_{m}": lo,
            f"conf_int_upper_{m}": hi,
            f"r_squared_{m}": r2,
            f"n_years_{m}": n.astype(np.int32),
        }
    )


# ----------------------------------------------------------------------------
# Public entry point (same signature as before, plus new options)
# ----------------------------------------------------------------------------


def compute_trend_stats(
    *,
    results_root: str | os.PathLike[str],
    potato_grid_file: str | os.PathLike[str],
    output_stats_parquet_path: str | os.PathLike[str],
    output_yearly_data_path: str | os.PathLike[str] | None = None,
    year_range: tuple = (1981, 2020),
    min_years: int = 10,
    aggregate_by: str = "season",
    exclude_truncated: bool = True,
    workers: int = 8,
    # kept for backwards compatibility with main.py; no longer used
    batch_size: int | None = None,
    n_workers: int | None = None,
    threads_per_worker: int | None = None,
    memory_limit: str | None = None,
) -> pd.DataFrame:
    """
    Compute OLS trend stats for APP/ABU/AFU across years for each grid point.

    Reads every `simcast_results.parquet` under `results_root` once, builds a
    yearly table (one row per ID and season), fits ``metric ~ year`` per ID and
    writes the stats parquet (one row per potato-grid ID; IDs without results
    have NaN statistics).  ``year_range`` is inclusive.
    """
    if n_workers is not None and workers == 8:
        workers = n_workers

    yearly = build_yearly_table(
        results_root,
        aggregate_by=aggregate_by,
        exclude_truncated=exclude_truncated,
        workers=workers,
    )
    if output_yearly_data_path is not None:
        Path(output_yearly_data_path).parent.mkdir(parents=True, exist_ok=True)
        yearly.to_parquet(output_yearly_data_path, index=False)

    potato_grid = gpd.read_file(os.fspath(potato_grid_file))
    if "ID" not in potato_grid.columns:
        potato_grid["ID"] = potato_grid.index
    stats_df = pd.DataFrame({"ID": potato_grid["ID"].to_numpy()})

    for metric in METRICS:
        st = ols_trend_by_id(yearly, metric, min_years=min_years, year_range=year_range)
        stats_df = stats_df.merge(st, on="ID", how="left")

    Path(output_stats_parquet_path).parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_parquet(output_stats_parquet_path, index=False)

    n_fit = int(stats_df["slope_app"].notna().sum())
    print(f"[trend] {n_fit} / {len(stats_df)} IDs fitted "
          f"({yearly['year'].min()}..{yearly['year'].max()}, {yearly['year'].nunique()} seasons)")
    return stats_df


# ----------------------------------------------------------------------------
# Maps
# ----------------------------------------------------------------------------


def _load_trend_gdf(shapefile_path, stats_parquet_path, variable_prefix):
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
    return potato_grid.merge(stats_df, on="ID", how="left"), p_value_col, slope_col


def _plot_admin0(ax, gadm_peru_gpkg, crs):
    if gadm_peru_gpkg is None:
        return
    try:
        from animation_all_lands import load_gadm_peru

        gadm_admin0, _ = load_gadm_peru(gadm_peru_gpkg)
        gadm_admin0.to_crs(crs).boundary.plot(ax=ax, edgecolor="black", linewidth=1.2, zorder=10)
    except Exception as e:
        print(f"Could not load GADM boundaries: {e}")


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

    trend_gdf, p_value_col, slope_col = _load_trend_gdf(
        shapefile_path, stats_parquet_path, variable_prefix
    )

    # 0=NoData, 1=NotSignificant, 2=SignificantIncrease, 3=SignificantDecrease
    trend_gdf["trend_type"] = 0
    valid_mask = trend_gdf[p_value_col].notna()
    trend_gdf.loc[valid_mask & (trend_gdf[p_value_col] >= p_threshold), "trend_type"] = 1
    significant_mask = valid_mask & (trend_gdf[p_value_col] < p_threshold)
    trend_gdf.loc[significant_mask & (trend_gdf[slope_col] > 0), "trend_type"] = 2
    trend_gdf.loc[significant_mask & (trend_gdf[slope_col] < 0), "trend_type"] = 3

    counts = trend_gdf["trend_type"].value_counts()
    print(f"[trend] {variable_prefix}: increase={counts.get(2, 0)}, decrease={counts.get(3, 0)}, "
          f"not significant={counts.get(1, 0)}, no data={counts.get(0, 0)}")

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    colors = {
        2: "#D94535",  # Increase
        3: "#009B77",  # Decrease
        1: "#E6E7E8",  # Not significant
        0: "#FFFFFF00",  # Transparent
    }
    trend_gdf["color"] = trend_gdf["trend_type"].map(colors)
    trend_gdf.plot(color=trend_gdf["color"], ax=ax, markersize=markersize)
    _plot_admin0(ax, gadm_peru_gpkg, trend_gdf.crs)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label=label, markerfacecolor=colors[k], markersize=10)
        for k, label in ((2, "Significant Increase"), (3, "Significant Decrease"), (1, "No Significant Trend"))
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        title=f"{variable_prefix.upper()} Trend (p < {p_threshold})",
        fontsize=12,
    )
    ax.set_title(f"Significance of {variable_prefix.upper()} Trends", fontsize=16, pad=20)
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

    trend_gdf, p_value_col, slope_col = _load_trend_gdf(
        shapefile_path, stats_parquet_path, variable_prefix
    )

    significant_mask = trend_gdf[p_value_col].notna() & (trend_gdf[p_value_col] < p_threshold)
    significant_gdf = trend_gdf[significant_mask]
    nonsignificant_gdf = trend_gdf[~significant_mask]

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
        significant_gdf.plot(
            ax=ax,
            column=slope_col,
            cmap=custom_cmap,
            markersize=markersize,
            vmin=-max_abs_slope,
            vmax=max_abs_slope,
            legend=True,
            legend_kwds={
                "label": f"{variable_prefix.upper()} Trend Coefficient",
                "orientation": "horizontal",
                "pad": 0.01,
            },
        )
    else:
        print(f"No significant points found for {variable_prefix.upper()} at p < {p_threshold}")

    _plot_admin0(ax, gadm_peru_gpkg, trend_gdf.crs)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="No Significant Trend",
               markerfacecolor="#E6E7E8", markersize=10)
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    ax.set_title(f"Magnitude of {variable_prefix.upper()} Trends (p < {p_threshold})", fontsize=16, pad=20)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    Path(output_image_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_image_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
