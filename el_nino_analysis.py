"""
Fungicide applications (APP) in El Niño years vs. the non-Niño average.

Inputs
------
* Merged yearly APP rasters ``<results_root>/<year>/APP_<variety>_<year>.tif``
  (see ``merge_rasters.py``).
* A table of El Niño events (``events_csv``) with columns
  ``start`` (YYYY-MM), ``end`` (YYYY-MM), ``magnitude`` and ``source``.

SIMCAST year
------------
Year ``Y`` groups the three seasons that start in ``Y``: Sierra Media
(cultivo3, Apr-Jul Y), Costa y Valle Interandino (cultivo2, Jul-Dec Y) and
Sierra Alta (cultivo1, Nov Y - Apr Y+1).  Year ``Y`` is an El Niño year when
an event overlaps at least ``min_overlap_months`` months of any of those
seasons; its magnitude is the strongest overlapping event.

Statistics
----------
* Reference ("promedio sin Niño"): all non-Niño years pooled.  The map shows
  the per-pixel mean; the KL reference distribution is the pooled
  distribution of yearly APP values (APP is an integer count, so the
  distributions are histograms over 0, 1, 2, ...).
* ``KL(P_year || Q_ref)`` in nats, with add-one (Laplace) smoothing of the
  counts so empty bins do not give infinite values.
* Only pixels valid in every analysed year are used, so all distributions
  describe the same set of points.  Incomplete years (e.g. 2020, which has
  no Sierra Alta season) are excluded.
* To read the KL values, the same KL is computed for every non-Niño year
  against the reference built from the *other* non-Niño years
  (leave-one-out); this gives the year-to-year spread expected without
  El Niño.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

SEASONS = {  # cultivo: (region, first month offset, last month offset) relative to Jan of Y
    "cultivo3": ("Sierra Media", 3, 6),  # Apr..Jul Y
    "cultivo2": ("Costa y Valle Interandino", 6, 11),  # Jul..Dec Y
    "cultivo1": ("Sierra Alta", 10, 15),  # Nov Y .. Apr Y+1
}

MAGNITUDE_ORDER = ["débil", "moderado", "fuerte", "extraordinario"]
MAGNITUDE_LABELS = {
    "débil": "Débil",
    "moderado": "Moderado",
    "fuerte": "Fuerte",
    "extraordinario": "Extraordinario\n(mega Niño)",
}


# ----------------------------------------------------------------------------
# Events -> SIMCAST years
# ----------------------------------------------------------------------------


def load_events(events_csv: str | os.PathLike[str]) -> pd.DataFrame:
    ev = pd.read_csv(events_csv, comment="#")
    ev["start"] = pd.PeriodIndex(ev["start"], freq="M")
    ev["end"] = pd.PeriodIndex(ev["end"], freq="M")
    ev["magnitude"] = ev["magnitude"].str.strip().str.lower()
    bad = set(ev["magnitude"]) - set(MAGNITUDE_ORDER)
    if bad:
        raise ValueError(f"unknown magnitudes {bad}; use {MAGNITUDE_ORDER}")
    return ev


def _months(start: pd.Period, end: pd.Period) -> set[pd.Period]:
    return set(pd.period_range(start, end, freq="M"))


def classify_years(
    events: pd.DataFrame,
    years,
    *,
    min_overlap_months: int = 3,
) -> pd.DataFrame:
    """
    One row per SIMCAST year: overlap months of El Niño with each season,
    ``is_nino`` and ``magnitude`` (strongest overlapping event).
    """
    rows = []
    for y in years:
        jan = pd.Period(f"{y}-01", freq="M")
        row = {"year": y, "is_nino": False, "magnitude": None, "events": ""}
        best = -1
        names = []
        for cultivo, (region, a, b) in SEASONS.items():
            season = _months(jan + a, jan + b)
            n_tot = 0
            for _, e in events.iterrows():
                n = len(season & _months(e["start"], e["end"]))
                if n >= min_overlap_months:
                    row["is_nino"] = True
                    rank = MAGNITUDE_ORDER.index(e["magnitude"])
                    if rank > best:
                        best, row["magnitude"] = rank, e["magnitude"]
                    names.append(f"{e['start']}..{e['end']}")
                n_tot = max(n_tot, n)
            row[f"overlap_{region}"] = n_tot
        row["events"] = "; ".join(sorted(set(names)))
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Rasters
# ----------------------------------------------------------------------------


@dataclass
class AppStack:
    years: list[int]
    data: np.ndarray  # (n_years, n_pixels) int, only common valid pixels
    rows: np.ndarray
    cols: np.ndarray
    shape: tuple[int, int]
    extent: list[float]
    crs: object


def load_app_stack(results_root, variety: str, years) -> AppStack:
    """Read the merged APP rasters and keep the pixels valid in every year."""
    root = Path(os.fspath(results_root))
    arrays, ok_years = [], []
    for y in years:
        p = root / str(y) / f"APP_{variety}_{y}.tif"
        if not p.exists():
            raise FileNotFoundError(p)
        with rasterio.open(p) as src:
            arrays.append(src.read(1, masked=True))
            shape = src.shape
            b = src.bounds
            extent = [b.left, b.right, b.bottom, b.top]
            crs = src.crs
        ok_years.append(y)
    stack = np.ma.stack(arrays)
    common = ~np.ma.getmaskarray(stack).any(axis=0)
    rows, cols = np.nonzero(common)
    data = np.asarray(stack[:, rows, cols]).astype(np.int64)
    return AppStack(ok_years, data, rows, cols, shape, extent, crs)


def to_image(stack: AppStack, values: np.ndarray) -> np.ma.MaskedArray:
    img = np.full(stack.shape, np.nan)
    img[stack.rows, stack.cols] = values
    return np.ma.masked_invalid(img)


# ----------------------------------------------------------------------------
# KL divergence
# ----------------------------------------------------------------------------


def histogram(values: np.ndarray, n_bins: int) -> np.ndarray:
    return np.bincount(values.ravel(), minlength=n_bins)[:n_bins].astype(np.float64)


def kl_divergence(p_counts: np.ndarray, q_counts: np.ndarray, *, alpha: float = 1.0) -> float:
    """KL(P || Q) in nats from two count histograms, with add-``alpha`` smoothing."""
    p = p_counts + alpha
    q = q_counts + alpha
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def analyse(
    stack: AppStack,
    years_table: pd.DataFrame,
    *,
    alpha: float = 1.0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    KL of every year vs. the non-Niño reference.

    Returns (table, reference mean per pixel).  For non-Niño years the
    reference excludes the year itself (leave-one-out).
    """
    yt = years_table.set_index("year").loc[stack.years]
    nino = yt["is_nino"].to_numpy()
    n_bins = int(stack.data.max()) + 1
    hists = np.stack([histogram(stack.data[i], n_bins) for i in range(len(stack.years))])
    ref_counts = hists[~nino].sum(axis=0)
    ref_mean = stack.data[~nino].mean(axis=0)

    rows = []
    for i, y in enumerate(stack.years):
        q = ref_counts - hists[i] if not nino[i] else ref_counts
        rows.append(
            {
                "year": y,
                "is_nino": bool(nino[i]),
                "magnitude": yt.loc[y, "magnitude"],
                "events": yt.loc[y, "events"],
                "kl_nats": kl_divergence(hists[i], q, alpha=alpha),
                "mean_app": float(stack.data[i].mean()),
                "ref_mean_app": float(ref_mean.mean()),
                "delta_mean_app": float(stack.data[i].mean() - ref_mean.mean()),
                **{c: yt.loc[y, c] for c in yt.columns if c.startswith("overlap_")},
            }
        )
    return pd.DataFrame(rows), ref_mean


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------


def plot_nino_maps(
    stack: AppStack,
    table: pd.DataFrame,
    ref_mean: np.ndarray,
    *,
    title: str,
    out_path: str | os.PathLike[str],
    gadm_peru_gpkg=None,
    cmap: str = "PuRd",
    ncols_max: int = 12,
    min_magnitude: str = "moderado",
    anomaly: bool = False,
):
    """
    Reference map + one Peru map per El Niño year (magnitude >= ``min_magnitude``),
    left to right in time, with the magnitude and the KL divergence below each map.
    """
    import matplotlib.pyplot as plt

    from matplotlib.colors import TwoSlopeNorm

    rank = MAGNITUDE_ORDER.index(min_magnitude)
    strong = table["magnitude"].map(lambda m: m in MAGNITUDE_ORDER and MAGNITUDE_ORDER.index(m) >= rank)
    nino = table[table["is_nino"] & strong].sort_values("year")
    non = table[~table["is_nino"]]
    panels = [("Promedio sin Niño", ref_mean, None)]
    for _, r in nino.iterrows():
        i = stack.years.index(int(r["year"]))
        vals = stack.data[i] - ref_mean if anomaly else stack.data[i]
        panels.append((str(int(r["year"])), vals, r))

    vmax = float(np.percentile(stack.data, 99))
    if anomaly:
        diffs = np.concatenate([np.abs(v) for _, v, r in panels if r is not None]) if len(panels) > 1 else np.array([1.0])
        dmax = max(float(np.percentile(diffs, 99)), 0.5)
        norm_anom = TwoSlopeNorm(vmin=-dmax, vcenter=0.0, vmax=dmax)
    n = len(panels)
    ncols = min(n, ncols_max)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 5.2 * nrows + 1.2), squeeze=False)

    adm0 = adm1 = None
    if gadm_peru_gpkg is not None:
        try:
            from animation_all_lands import load_gadm_peru

            adm0, adm1 = load_gadm_peru(gadm_peru_gpkg)
            if stack.crs is not None:
                adm0, adm1 = adm0.to_crs(stack.crs), adm1.to_crs(stack.crs)
        except Exception as e:
            print(f"[el_nino] no GADM boundaries: {e}")

    im = im_anom = None
    for k, ax in enumerate(axes.ravel()):
        ax.set_axis_off()
        if k >= n:
            continue
        name, values, r = panels[k]
        if anomaly and r is not None:
            im_anom = ax.imshow(to_image(stack, values), extent=stack.extent, origin="upper",
                                cmap="RdBu_r", norm=norm_anom, interpolation="nearest")
        else:
            im = ax.imshow(to_image(stack, values), extent=stack.extent, origin="upper",
                           cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        if adm1 is not None:
            adm1.boundary.plot(ax=ax, edgecolor="0.55", linewidth=0.3)
            adm0.boundary.plot(ax=ax, edgecolor="black", linewidth=0.6)
        ax.set_xlim(-81.3, -68.5)
        ax.set_ylim(-18.5, -3.3)  # potato area (about -18.3 .. -4.4)
        ax.set_title(name, fontsize=12, fontweight="bold")
        if r is None:
            sub = f"{len(non)} años sin Niño\nAPP medio {float(np.mean(values)):.2f}"
        else:
            sub = (f"{MAGNITUDE_LABELS.get(r['magnitude'], r['magnitude'])}\n"
                   f"KL = {r['kl_nats']:.3f} nats\n"
                   f"ΔAPP medio {r['delta_mean_app']:+.2f}")
        ax.text(0.5, -0.02, sub, transform=ax.transAxes, ha="center", va="top", fontsize=10)

    q = non["kl_nats"]
    fig.suptitle(
        f"{title}\nKL de años sin Niño (cada uno vs. los demás): mediana {q.median():.3f}, "
        f"máx {q.max():.3f} nats",
        fontsize=12,
    )
    if im is not None and not anomaly:
        cax = fig.add_axes([0.3, 0.06, 0.4, 0.025])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="max")
        cb.set_label("Aplicaciones de fungicida por campaña (APP)")
    if anomaly:
        if im is not None:
            cax = fig.add_axes([0.02, 0.06, 0.12, 0.025])
            cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="max")
            cb.set_label("APP promedio sin Niño")
        if im_anom is not None:
            cax = fig.add_axes([0.35, 0.06, 0.4, 0.025])
            cb = fig.colorbar(im_anom, cax=cax, orientation="horizontal", extend="both")
            cb.set_label("Año Niño − promedio sin Niño (aplicaciones por campaña)")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.2, wspace=0.04, hspace=0.6)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def run_el_nino_analysis(
    *,
    results_root,
    variety: str,
    variety_label: str,
    events_csv,
    years=range(1981, 2020),
    min_overlap_months: int = 3,
    min_magnitude: str = "moderado",
    gadm_peru_gpkg=None,
    out_dir=None,
) -> dict[str, str]:
    """Classify years, compute KL vs. the non-Niño reference, write CSV + figure."""
    out_dir = Path(os.fspath(out_dir or results_root)) / "el_nino"
    out_dir.mkdir(parents=True, exist_ok=True)
    events = load_events(events_csv)
    years = list(years)
    years_table = classify_years(events, years, min_overlap_months=min_overlap_months)
    stack = load_app_stack(results_root, variety, years)
    table, ref_mean = analyse(stack, years_table)
    table.insert(1, "variety", variety)
    csv_path = out_dir / f"kl_app_el_nino_{variety}.csv"
    table.round(4).to_csv(csv_path, index=False)
    png_path = out_dir / f"app_el_nino_vs_promedio_{variety}.png"
    plot_nino_maps(
        stack,
        table,
        ref_mean,
        title=(f"Aplicaciones de fungicida (APP), variedad {variety_label}: años El Niño costero "
               f"(ENFEN 2024, magnitud ≥ {min_magnitude}) vs. promedio sin Niño, "
               f"{years[0]}-{years[-1]}"),
        out_path=png_path,
        gadm_peru_gpkg=gadm_peru_gpkg,
        min_magnitude=min_magnitude,
    )
    anom_path = out_dir / f"app_el_nino_anomalia_{variety}.png"
    plot_nino_maps(
        stack,
        table,
        ref_mean,
        title=(f"Anomalía de aplicaciones (APP), variedad {variety_label}: año El Niño costero "
               f"− promedio sin Niño, {years[0]}-{years[-1]}"),
        out_path=anom_path,
        gadm_peru_gpkg=gadm_peru_gpkg,
        min_magnitude=min_magnitude,
        anomaly=True,
    )
    print(f"[el_nino] {variety}: {int(table.is_nino.sum())} Niño years, "
          f"{int((~table.is_nino).sum())} reference years, {stack.data.shape[1]} pixels -> {png_path}")
    return {"csv": os.fspath(csv_path), "png": os.fspath(png_path), "anomaly_png": os.fspath(anom_path)}
