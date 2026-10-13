"""
SIMCAST outputs (APP, BU, FU) in El Niño years vs. the non-Niño average.

Inputs
------
* Merged yearly rasters ``<results_root>/<year>/<VAR>_<variety>_<year>.tif``
  with VAR in APP, BU, FU (see ``merge_rasters.py``).
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
  distribution of the yearly values.  Distributions are histograms with a
  fixed bin width per variable (``VARIABLES``): 1 application for APP
  (integer counts), 10 units for the season sums BU and FU (values up to
  ~1000).
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

# variable -> (bin width for the KL histograms, short label, long label, colormap)
VARIABLES = {
    "APP": (1.0, "aplicaciones", "Aplicaciones de fungicida por campaña (APP)", "PuRd"),
    "BU": (10.0, "BU", "Blight Units, suma por campaña (BU)", "YlOrRd"),
    "FU": (10.0, "FU", "Fungicide Units, suma por campaña (FU)", "YlGnBu"),
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
class ValueStack:
    years: list[int]
    data: np.ndarray  # (n_years, n_pixels), only common valid pixels
    rows: np.ndarray
    cols: np.ndarray
    shape: tuple[int, int]
    extent: list[float]
    crs: object


def load_stack(results_root, variety: str, years, variable: str = "APP") -> ValueStack:
    """Read the merged yearly rasters of one variable; keep pixels valid in every year."""
    root = Path(os.fspath(results_root))
    arrays, ok_years = [], []
    for y in years:
        p = root / str(y) / f"{variable}_{variety}_{y}.tif"
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
    data = np.asarray(stack[:, rows, cols])
    if np.allclose(data, np.round(data)):
        data = np.round(data).astype(np.int64)
    return ValueStack(ok_years, data, rows, cols, shape, extent, crs)


def load_app_stack(results_root, variety: str, years) -> ValueStack:
    return load_stack(results_root, variety, years, "APP")


AppStack = ValueStack  # backwards-compatible name


def to_image(stack: ValueStack, values: np.ndarray) -> np.ma.MaskedArray:
    img = np.full(stack.shape, np.nan)
    img[stack.rows, stack.cols] = values
    return np.ma.masked_invalid(img)


# ----------------------------------------------------------------------------
# KL divergence
# ----------------------------------------------------------------------------


def histogram(values: np.ndarray, n_bins: int, *, bin_width: float = 1.0) -> np.ndarray:
    """Counts in bins [k*w, (k+1)*w) for k = 0..n_bins-1 (values must be >= 0)."""
    idx = np.floor(np.asarray(values, dtype=np.float64).ravel() / bin_width).astype(np.int64)
    return np.bincount(np.clip(idx, 0, n_bins - 1), minlength=n_bins)[:n_bins].astype(np.float64)


def kl_divergence(p_counts: np.ndarray, q_counts: np.ndarray, *, alpha: float = 1.0) -> float:
    """KL(P || Q) in nats from two count histograms, with add-``alpha`` smoothing."""
    p = p_counts + alpha
    q = q_counts + alpha
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def analyse(
    stack: ValueStack,
    years_table: pd.DataFrame,
    *,
    alpha: float = 1.0,
    variable: str = "APP",
    bin_width: float | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    KL of every year vs. the non-Niño reference.

    Returns (table, reference mean per pixel).  For non-Niño years the
    reference excludes the year itself (leave-one-out).
    """
    yt = years_table.set_index("year").loc[stack.years]
    nino = yt["is_nino"].to_numpy()
    w = bin_width if bin_width is not None else VARIABLES.get(variable, (1.0,))[0]
    n_bins = int(np.floor(float(stack.data.max()) / w)) + 1
    hists = np.stack([histogram(stack.data[i], n_bins, bin_width=w) for i in range(len(stack.years))])
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
                f"mean_{variable.lower()}": float(stack.data[i].mean()),
                f"ref_mean_{variable.lower()}": float(ref_mean.mean()),
                f"delta_mean_{variable.lower()}": float(stack.data[i].mean() - ref_mean.mean()),
                "kl_bin_width": w,
                **{c: yt.loc[y, c] for c in yt.columns if c.startswith("overlap_")},
            }
        )
    return pd.DataFrame(rows), ref_mean


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------


def plot_nino_maps(
    stack: ValueStack,
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
    variable: str = "APP",
):
    """
    Reference map + one Peru map per El Niño year (magnitude >= ``min_magnitude``),
    left to right in time, with the magnitude and the KL divergence below each map.
    """
    import matplotlib.pyplot as plt

    rank = MAGNITUDE_ORDER.index(min_magnitude)
    strong = table["magnitude"].map(lambda m: m in MAGNITUDE_ORDER and MAGNITUDE_ORDER.index(m) >= rank)
    nino = table[table["is_nino"] & strong].sort_values("year")
    non = table[~table["is_nino"]]
    panels = [("Promedio sin Niño", ref_mean, None)]
    for _, r in nino.iterrows():
        i = stack.years.index(int(r["year"]))
        vals = stack.data[i] - ref_mean if anomaly else stack.data[i]
        panels.append((str(int(r["year"])), vals, r))

    from viz_palette import diverging_classes, sequential_classes

    # discrete classes, identical for every panel of the figure
    seq = sequential_classes(np.concatenate([stack.data.ravel(), np.ravel(ref_mean)]))
    seq_cmap, seq_norm = seq.cmap_norm()
    if anomaly:
        diffs = np.concatenate([v for _, v, r in panels if r is not None]) if len(panels) > 1 else np.array([1.0])
        div = diverging_classes(diffs)
        div_cmap, div_norm = div.cmap_norm()
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
                                cmap=div_cmap, norm=div_norm, interpolation="nearest")
        else:
            im = ax.imshow(to_image(stack, values), extent=stack.extent, origin="upper",
                           cmap=seq_cmap, norm=seq_norm, interpolation="nearest")
        if adm1 is not None:
            adm1.boundary.plot(ax=ax, edgecolor="0.55", linewidth=0.3)
            adm0.boundary.plot(ax=ax, edgecolor="black", linewidth=0.6)
        ax.set_xlim(-81.3, -68.5)
        ax.set_ylim(-18.5, -3.3)  # potato area (about -18.3 .. -4.4)
        ax.set_title(name, fontsize=12, fontweight="bold")
        if r is None:
            sub = f"{len(non)} años sin Niño\n{variable} medio {float(np.mean(values)):.2f}"
        else:
            sub = (f"{MAGNITUDE_LABELS.get(r['magnitude'], r['magnitude'])}\n"
                   f"KL = {r['kl_nats']:.3f} nats\n"
                   f"Δ{variable} medio {r[f'delta_mean_{variable.lower()}']:+.2f}")
        ax.text(0.5, -0.02, sub, transform=ax.transAxes, ha="center", va="top", fontsize=10)

    q = non["kl_nats"]
    fig.suptitle(
        f"{title}\nKL de años sin Niño (cada uno vs. los demás): mediana {q.median():.3f}, "
        f"máx {q.max():.3f} nats",
        fontsize=12,
    )
    if im is not None and not anomaly:
        cax = fig.add_axes([0.3, 0.06, 0.4, 0.025])
        seq.colorbar(fig, im, cax, label=VARIABLES.get(variable, (None, None, variable))[2])
    if anomaly:
        if im is not None:
            cax = fig.add_axes([0.02, 0.06, 0.16, 0.025])
            seq.colorbar(fig, im, cax, label=f"{variable} promedio sin Niño")
        if im_anom is not None:
            cax = fig.add_axes([0.35, 0.06, 0.4, 0.025])
            div.colorbar(fig, im_anom, cax,
                         label=f"Año Niño − promedio sin Niño ({VARIABLES.get(variable, (None, variable))[1]} por campaña)")
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
    variable: str = "APP",
) -> dict[str, str]:
    """Classify years, compute KL vs. the non-Niño reference, write CSV + figures."""
    out_dir = Path(os.fspath(out_dir or results_root)) / "el_nino"
    out_dir.mkdir(parents=True, exist_ok=True)
    events = load_events(events_csv)
    years = list(years)
    years_table = classify_years(events, years, min_overlap_months=min_overlap_months)
    var = variable.upper()
    v = var.lower()
    _, _, long_label, cmap = VARIABLES[var]
    stack = load_stack(results_root, variety, years, var)
    table, ref_mean = analyse(stack, years_table, variable=var)
    table.insert(1, "variety", variety)
    table.insert(2, "variable", var)
    csv_path = out_dir / f"kl_{v}_el_nino_{variety}.csv"
    table.round(4).to_csv(csv_path, index=False)
    png_path = out_dir / f"{v}_el_nino_vs_promedio_{variety}.png"
    plot_nino_maps(
        stack,
        table,
        ref_mean,
        title=(f"{long_label}, variedad {variety_label}: años El Niño costero "
               f"(ENFEN 2024, magnitud ≥ {min_magnitude}) vs. promedio sin Niño, "
               f"{years[0]}-{years[-1]}"),
        out_path=png_path,
        gadm_peru_gpkg=gadm_peru_gpkg,
        min_magnitude=min_magnitude,
        variable=var,
        cmap=cmap,
    )
    anom_path = out_dir / f"{v}_el_nino_anomalia_{variety}.png"
    plot_nino_maps(
        stack,
        table,
        ref_mean,
        title=(f"Anomalía — {long_label}, variedad {variety_label}: año El Niño costero "
               f"− promedio sin Niño, {years[0]}-{years[-1]}"),
        out_path=anom_path,
        gadm_peru_gpkg=gadm_peru_gpkg,
        min_magnitude=min_magnitude,
        anomaly=True,
        variable=var,
        cmap=cmap,
    )
    print(f"[el_nino] {variety} {var}: {int(table.is_nino.sum())} Niño years, "
          f"{int((~table.is_nino).sum())} reference years, {stack.data.shape[1]} pixels -> {png_path}")
    return {"csv": os.fspath(csv_path), "png": os.fspath(png_path), "anomaly_png": os.fspath(anom_path)}
