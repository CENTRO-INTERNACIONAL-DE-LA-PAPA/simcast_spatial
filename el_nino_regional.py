"""
El Niño vs. non-Niño years: point-wise anomalies, composites with
significance, and regional statistics (for a scientific audience).

Design (agreed Oct 2026)
------------------------
* Data: ``yearly_aggregated_data.parquet`` (one value per point and SIMCAST
  year: APP = applications, ABU = blight units, AFU = fungicide units, summed
  over the season).  Years 1981-2019 (2020 has no Sierra Alta season).
* Years: El Niño Costero events from the ENFEN 2024 list (see
  ``el_nino_analysis.classify_years``).  Reference = years with no El Niño at
  all (weak events are excluded from the reference and not plotted).
  Groups: ``extraordinario`` and ``fuerte + moderado``.
* Units: SIMCAST region (one season each) x latitude zone
  (Norte >= -9 deg, Centro -15..-9 deg, Sur < -15 deg).

Per point
---------
* Standardised anomaly ``z = (x_year - mean_ref) / sd_ref`` (sd over the
  reference years, floored at half a unit so points that never change, e.g.
  BU = 0 every year, do not blow up).
* Composite = mean z over the years of a group.
* Significance: permutation test of mean(group) - mean(reference) (labels
  shuffled among group + reference years; all relabellings enumerated when
  there are <= 20 000, e.g. 4 of 25 years, else 20 000 random ones),
  two-sided.  Maps stipple local p <= 0.05.  Field significance (is the
  share of locally significant points larger than chance?) by the
  Livezey & Chen (1983, MWR 111: 46-59) relabelling test, applying the same
  relabelling to every point so spatial correlation is preserved.  A
  point-wise Benjamini-Hochberg FDR mask (alpha 0.10, Wilks 2016) is also
  stored (``sig_fdr``); with 4-6 El Niño years it is very conservative.

Per unit and year
-----------------
* ``delta``: regional mean minus the reference mean (variable units) and %.
* ``z_region`` and ``pct_rank``: the anomaly against the reference years'
  own anomalies, each computed leave-one-out (that year vs. the other
  reference years) -> the natural year-to-year spread.
* ``js_bits``: Jensen-Shannon divergence (base 2, 0..1) between the year's
  distribution of point values and the pooled reference distribution; for BU
  only the positive values (``zero_frac`` columns carry the zero share).
* ``wasserstein``: 1-D Wasserstein (earth mover's) distance, in variable
  units; exact for these integer variables (bins of 1).
* JS and Wasserstein are also given as percentiles against the reference
  years (leave-one-out), because both grow with sampling noise.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from el_nino_analysis import MAGNITUDE_LABELS, classify_years, load_events

VARIABLES = {  # column -> (short name, unit label, JS bin width, sd floor)
    "APP": ("APP", "aplicaciones", 1.0, 0.5),
    "ABU": ("BU", "blight units", 10.0, 0.5),
    "AFU": ("FU", "fungicide units", 10.0, 0.5),
}
ZONES = [("Norte", -9.0, 90.0), ("Centro", -15.0, -9.0), ("Sur", -90.0, -15.0)]
REGION_ORDER = ["Costa y Valle Interandino", "Sierra Media", "Sierra Alta"]
REGION_SHORT = {"Costa y Valle Interandino": "Costa/VI", "Sierra Media": "S. Media", "Sierra Alta": "S. Alta"}
GROUPS = {
    "extraordinario": ("extraordinario",),
    "fuerte+moderado": ("fuerte", "moderado"),
}
GROUP_LABELS = {"extraordinario": "Extraordinario (mega Niño)", "fuerte+moderado": "Fuerte + moderado"}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------


def zone_of(lat: np.ndarray) -> np.ndarray:
    out = np.empty(len(lat), dtype=object)
    for name, lo, hi in ZONES:
        out[(lat >= lo) & (lat < hi)] = name
    return out


def load_matrix(yearly_path, years) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Points table (ID, lon, lat, region, zone) and {var: (n_points, n_years) array}."""
    y = pd.read_parquet(yearly_path, columns=["ID", "lon", "lat", "region", "year", *VARIABLES])
    y = y[y["year"].isin(years)]
    pts = y.drop_duplicates("ID")[["ID", "lon", "lat", "region"]].sort_values("ID").reset_index(drop=True)
    pts["zone"] = zone_of(pts["lat"].to_numpy())
    pts["unit"] = pts["region"].map(REGION_SHORT) + " · " + pts["zone"]
    row = pd.Series(np.arange(len(pts)), index=pts["ID"].to_numpy())
    col = pd.Series(np.arange(len(years)), index=list(years))
    r = row.loc[y["ID"].to_numpy()].to_numpy()
    c = col.loc[y["year"].to_numpy()].to_numpy()
    mats = {}
    for v in VARIABLES:
        m = np.full((len(pts), len(years)), np.nan, dtype=np.float64)
        m[r, c] = y[v].to_numpy(dtype=np.float64)
        mats[v] = m
    return pts, mats


# ----------------------------------------------------------------------------
# Point-wise statistics
# ----------------------------------------------------------------------------


def permutation_pvalues(
    x: np.ndarray, group_cols: np.ndarray, ref_cols: np.ndarray, *,
    n_perm: int = 20_000, seed: int = 0, chunk: int = 2_000, exact_max: int = 20_000,
    return_field: bool = False, local_alpha: float = 0.05,
):
    """
    Two-sided permutation p-value of mean(group) - mean(ref), per row.

    All C(n, k) relabellings are enumerated when there are at most
    ``exact_max`` of them (exact test, p = share of relabellings at least as
    extreme, including the observed one); otherwise ``n_perm`` random
    relabellings are used (p = (exceed + 1) / (n_perm + 1)).  The resolution
    matters: with ~3e5 points, the FDR correction needs p-values well below
    1e-3.
    """
    from itertools import combinations
    from math import comb

    cols = np.r_[group_cols, ref_cols]
    data = x[:, cols].astype(np.float32)
    n, k = len(cols), len(group_cols)
    exact = comb(n, k) <= exact_max
    if exact:
        combos = list(combinations(range(n), k))
        sel = np.zeros((n, len(combos)), dtype=np.float32)
        for j, c in enumerate(combos):
            sel[list(c), j] = 1.0
    else:
        rng = np.random.default_rng(seed)
        sel = np.zeros((n, n_perm), dtype=np.float32)
        for j in range(n_perm):
            sel[rng.choice(n, size=k, replace=False), j] = 1.0
    n_rel = sel.shape[1]
    p = np.empty(len(x))
    # field significance (Livezey & Chen 1983): for every relabelling, the
    # number of points that would be locally significant (|diff| in the top
    # ``local_alpha`` of that point's own relabelling distribution).  The same
    # relabelling is applied to the whole map, so spatial correlation is kept.
    field_counts = np.zeros(n_rel, dtype=np.int64)
    for a in range(0, len(x), chunk):
        d = data[a:a + chunk]
        tot = d.sum(axis=1, keepdims=True)
        obs = d[:, :k].mean(axis=1, keepdims=True) - d[:, k:].mean(axis=1, keepdims=True)
        s = d @ sel
        perm = np.abs(s / k - (tot - s) / (n - k))
        exceed = (perm >= np.abs(obs) - 1e-4).sum(axis=1)
        p[a:a + chunk] = exceed / n_rel if exact else (exceed + 1) / (n_rel + 1)
        if return_field:
            q = np.quantile(perm, 1 - local_alpha, axis=1, method="higher")[:, None]
            constant = (perm.max(axis=1, keepdims=True) <= 1e-6)
            field_counts += ((perm >= q - 1e-4) & ~constant).sum(axis=0)
    if return_field:
        return p, field_counts / len(x)
    return p


def fdr_mask(p: np.ndarray, alpha: float = 0.10) -> np.ndarray:
    """Benjamini-Hochberg: True where the null is rejected at FDR ``alpha``."""
    order = np.argsort(p)
    m = len(p)
    thresh = alpha * np.arange(1, m + 1) / m
    ok = p[order] <= thresh
    if not ok.any():
        return np.zeros(m, dtype=bool)
    kmax = np.max(np.nonzero(ok)[0])
    out = np.zeros(m, dtype=bool)
    out[order[: kmax + 1]] = True
    return out


def composites(x, years, groups_years, ref_years, *, sd_floor, n_perm=20_000, alpha_fdr=0.10,
               local_alpha=0.05):
    """{group: dict(z=..., delta=..., p=..., sig=...)} per point."""
    yi = {y: i for i, y in enumerate(years)}
    ref = np.array([yi[y] for y in ref_years])
    mu = x[:, ref].mean(axis=1)
    sd = np.maximum(x[:, ref].std(axis=1, ddof=1), sd_floor)
    out = {}
    for g, gy in groups_years.items():
        cols = np.array([yi[y] for y in gy])
        z = ((x[:, cols] - mu[:, None]) / sd[:, None]).mean(axis=1)
        delta = x[:, cols].mean(axis=1) - mu
        p, field = permutation_pvalues(x, cols, ref, n_perm=n_perm, return_field=True, local_alpha=local_alpha)
        constant = np.all(x[:, np.r_[cols, ref]] == x[:, [ref[0]]], axis=1)
        p[constant] = 1.0
        local = p <= local_alpha
        share = float(local.mean())
        out[g] = {
            "z": z, "delta": delta, "p": p, "years": list(gy),
            "sig": local,  # stippling: local p <= local_alpha
            "sig_fdr": fdr_mask(p, alpha_fdr),
            "share_local": share,
            "field_p": float(np.mean(field >= share - 1e-12)),
            "field_null_median": float(np.median(field)),
        }
    return out


# ----------------------------------------------------------------------------
# Regional statistics
# ----------------------------------------------------------------------------


def _hist(values: np.ndarray, width: float, n_bins: int) -> np.ndarray:
    idx = np.clip(np.floor(values / width).astype(np.int64), 0, n_bins - 1)
    return np.bincount(idx, minlength=n_bins).astype(np.float64)


def js_bits(p_counts: np.ndarray, q_counts: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits (0..1)."""
    if p_counts.sum() == 0 or q_counts.sum() == 0:
        return np.nan
    p = p_counts / p_counts.sum()
    q = q_counts / q_counts.sum()
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pm = np.nansum(np.where(p > 0, p * np.log2(p / m), 0.0))
        kl_qm = np.nansum(np.where(q > 0, q * np.log2(q / m), 0.0))
    return float(0.5 * (kl_pm + kl_qm))


def wasserstein_1(p_counts: np.ndarray, q_counts: np.ndarray, width: float = 1.0) -> float:
    """W1 between two histograms on the same bins (area between the CDFs)."""
    p = np.cumsum(p_counts / p_counts.sum())
    q = np.cumsum(q_counts / q_counts.sum())
    return float(np.abs(p - q).sum() * width)


def regional_table(pts, x, years, year_info, ref_years, *, var: str) -> pd.DataFrame:
    """One row per unit and year (Niño and reference years)."""
    short, unit_label, width, _ = VARIABLES[var]
    yi = {y: i for i, y in enumerate(years)}
    ref = [yi[y] for y in ref_years]
    rows = []
    for unit, g in pts.groupby("unit", sort=False):
        idx = g.index.to_numpy()
        xu = x[idx]  # (n_pts, n_years)
        n_bins = int(np.nanmax(xu) // 1) + 2
        vals_int = np.nan_to_num(xu).astype(np.int64)
        h1 = np.stack([np.bincount(vals_int[:, j], minlength=n_bins) for j in range(len(years))], axis=1).astype(float)
        # JS histograms: BU on positive values only (zeros reported apart)
        positive_only = var == "ABU"
        nb_js = int(np.nanmax(xu) // width) + 2
        hj = []
        for j in range(len(years)):
            v = xu[:, j]
            v = v[v > 0] if positive_only else v
            hj.append(_hist(v, width, nb_js))
        hj = np.stack(hj, axis=1)
        means = xu.mean(axis=0)
        zero = (xu == 0).mean(axis=0)

        def stats(j, ref_idx):
            mref = means[ref_idx].mean()
            return {
                "delta": means[j] - mref,
                "delta_pct": 100 * (means[j] - mref) / mref if mref else np.nan,
                "js_bits": js_bits(hj[:, j], hj[:, ref_idx].sum(axis=1)),
                "wasserstein": wasserstein_1(h1[:, j], h1[:, ref_idx].sum(axis=1)),
                "zero_frac": zero[j],
                "zero_frac_ref": zero[ref_idx].mean(),
                "ref_mean": mref,
            }

        loo = {r: stats(r, [q for q in ref if q != r]) for r in ref}
        ref_delta = np.array([loo[r]["delta"] for r in ref])
        ref_js = np.array([loo[r]["js_bits"] for r in ref])
        ref_w = np.array([loo[r]["wasserstein"] for r in ref])
        sd_ref = ref_delta.std(ddof=1)
        for y in years:
            j = yi[y]
            s = loo[j] if j in ref else stats(j, ref)
            info = year_info.loc[y]
            rows.append({
                "unit": unit,
                "region": g["region"].iloc[0],
                "zone": g["zone"].iloc[0],
                "variable": short,
                "units": unit_label,
                "year": y,
                "is_reference": j in ref,
                "magnitude": info["magnitude"] if info["is_nino"] else None,
                "n_points": len(idx),
                **s,
                "z_region": s["delta"] / sd_ref if sd_ref > 0 else np.nan,
                "pct_rank_delta": 100 * np.mean(ref_delta < s["delta"]),
                "p_delta": (np.sum(np.abs(ref_delta) >= abs(s["delta"])) + (0 if j in ref else 1))
                / (len(ref_delta) + (0 if j in ref else 1)),
                "pct_rank_js": 100 * np.mean(ref_js < s["js_bits"]),
                "pct_rank_wasserstein": 100 * np.mean(ref_w < s["wasserstein"]),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------


def _grid_image(grid, ids, values):
    img = np.full((grid["nrows"], grid["ncols"]), np.nan)
    pos = pd.Index(grid["ids"]).get_indexer(ids)
    img[grid["y_idx"][pos], grid["x_idx"][pos]] = values
    return img


def _stipple_points(grid, ids, sig, block: int = 10, min_share: float = 0.5):
    """Centres of blocks where most points are significant."""
    pos = pd.Index(grid["ids"]).get_indexer(ids)
    by, bx = grid["y_idx"][pos] // block, grid["x_idx"][pos] // block
    df = pd.DataFrame({"by": by, "bx": bx, "sig": sig})
    agg = df.groupby(["by", "bx"])["sig"].mean()
    agg = agg[agg >= min_share]
    t = grid["transform"]
    xs = t.c + (agg.index.get_level_values("bx").to_numpy() * block + block / 2) * t.a
    ys = t.f + (agg.index.get_level_values("by").to_numpy() * block + block / 2) * t.e
    return xs, ys


def plot_composites(pts, comps_by_var, grid, *, title, out_path, gadm_peru_gpkg=None):
    """Rows = variables, columns = groups; standardised anomaly + stippling."""
    import matplotlib.pyplot as plt

    from viz_palette import diverging_classes

    vars_ = list(comps_by_var)
    groups = list(next(iter(comps_by_var.values())))
    allz = np.concatenate([c[g]["z"] for c in comps_by_var.values() for g in groups])
    classes = diverging_classes(allz, n_per_arm=4)
    cmap, norm = classes.cmap_norm()

    adm0 = adm1 = None
    if gadm_peru_gpkg is not None:
        try:
            from animation_all_lands import load_gadm_peru

            adm0, adm1 = load_gadm_peru(gadm_peru_gpkg)
        except Exception as e:
            print(f"[el_nino_regional] no GADM: {e}")

    t = grid["transform"]
    extent = [t.c, t.c + grid["ncols"] * t.a, t.f + grid["nrows"] * t.e, t.f]
    fig, axes = plt.subplots(len(vars_), len(groups), figsize=(4.2 * len(groups), 5.0 * len(vars_) + 1.4),
                             squeeze=False)
    im = None
    for i, v in enumerate(vars_):
        for j, g in enumerate(groups):
            ax = axes[i, j]
            c = comps_by_var[v][g]
            im = ax.imshow(_grid_image(grid, pts["ID"].to_numpy(), c["z"]), extent=extent, origin="upper",
                           cmap=cmap, norm=norm, interpolation="nearest")
            sx, sy = _stipple_points(grid, pts["ID"].to_numpy(), c["sig"])
            ax.scatter(sx, sy, s=0.6, c="#0b0b0b", linewidths=0, alpha=0.8)
            if adm1 is not None:
                adm1.boundary.plot(ax=ax, edgecolor="0.6", linewidth=0.3)
                adm0.boundary.plot(ax=ax, edgecolor="black", linewidth=0.6)
            ax.set_xlim(-81.3, -68.5)
            ax.set_ylim(-18.5, -3.3)
            for lat in (-9, -15):
                ax.axhline(lat, color="0.35", lw=0.6, ls=(0, (4, 3)))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{VARIABLES[v][0]} · {GROUP_LABELS[g]}", fontsize=11, fontweight="bold")
            fp = c["field_p"]
            fp_txt = f"{fp:.3f}" if fp >= 0.001 else "< 0.001"
            ax.text(0.02, 0.02,
                    f"años: {', '.join(map(str, c['years']))}\n"
                    f"p < 0.05 local: {100 * c['share_local']:.1f} % de puntos "
                    f"(azar: ~{100 * c['field_null_median']:.0f} %)\n"
                    f"significancia de campo: p = {fp_txt}",
                    transform=ax.transAxes, fontsize=8, va="bottom")
    import textwrap

    fig.suptitle("\n".join(textwrap.wrap(title, 70)), fontsize=12)
    cax = fig.add_axes([0.2, 0.045, 0.6, 0.012])
    classes.colorbar(fig, im, cax, label="Anomalía estandarizada media (z, desviaciones estándar de los años sin Niño)")
    fig.text(0.5, 0.004, "Puntos negros: p < 0.05 local (prueba de permutación de años).\n"
             "Significancia de campo: Livezey y Chen (1983). Líneas discontinuas: límites de franja −9° y −15°.",
             ha="center", fontsize=8)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.09, wspace=0.03, hspace=0.12)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_dotbox(table, *, title, out_path):
    """Rows = variables; per unit: box of reference years, dots for El Niño years."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from viz_palette import RED

    mag_color = {"extraordinario": RED[650], "fuerte": RED[450], "moderado": RED[250]}
    units = [f"{REGION_SHORT[r]} · {z}" for r in REGION_ORDER for z, _, _ in ZONES]
    vars_ = list(dict.fromkeys(table["variable"]))
    fig, axes = plt.subplots(len(vars_), 1, figsize=(13, 3.6 * len(vars_) + 1.0), squeeze=False, sharex=True)
    rng = np.random.default_rng(0)
    for i, v in enumerate(vars_):
        ax = axes[i, 0]
        tv = table[table["variable"] == v]
        data = [tv[(tv["unit"] == u) & tv["is_reference"]]["z_region"].to_numpy() for u in units]
        ax.boxplot(data, positions=range(len(units)), widths=0.55, whis=(0, 100), showfliers=False,
                   patch_artist=True, boxprops=dict(facecolor="#e4e3de", edgecolor="#898781", lw=0.8),
                   medianprops=dict(color="#52514e", lw=1), whiskerprops=dict(color="#898781", lw=0.8),
                   capprops=dict(color="#898781", lw=0.8))
        nino = tv[tv["magnitude"].isin(list(mag_color))]
        for k, u in enumerate(units):
            nu = nino[nino["unit"] == u]
            jit = rng.uniform(-0.18, 0.18, len(nu))
            ax.scatter(k + jit, nu["z_region"], s=34, c=nu["magnitude"].map(mag_color),
                       edgecolors="white", linewidths=1.2, zorder=3)
            ref_lo = np.min(data[k]) if len(data[k]) else 0
            ref_hi = np.max(data[k]) if len(data[k]) else 0
            out = [(r["z_region"], dx, r["year"]) for (_, r), dx in zip(nu.iterrows(), jit)
                   if r["z_region"] > ref_hi or r["z_region"] < ref_lo]
            # place labels without overlap: walk outward from the box, min gap in z units
            span = np.diff(ax.get_ylim())[0] if ax.get_ylim()[1] > ax.get_ylim()[0] else 1.0
            gap = 0.045 * max(span, 4.0)
            for sign in (1, -1):
                side = sorted([o for o in out if (o[0] > 0) == (sign > 0)], key=lambda o: sign * o[0])
                last = None
                for zval, dx, yr in side:
                    ly = zval if last is None or sign * (zval - last) >= gap else last + sign * gap
                    last = ly
                    ax.annotate(str(yr), (k + dx, zval), xytext=(k + 0.27, ly), textcoords="data",
                                fontsize=7, va="center", color="#52514e",
                                arrowprops=dict(arrowstyle="-", color="#c3c2b7", lw=0.5) if abs(ly - zval) > 1e-9 else None)
        ax.axhline(0, color="#c3c2b7", lw=0.8, zorder=0)
        for b in (2.5, 5.5):
            ax.axvline(b, color="#e1e0d9", lw=1, zorder=0)
        ax.set_ylabel(f"{v}: anomalía regional (z)")
        ax.grid(axis="y", color="#e1e0d9", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[-1, 0].set_xticks(range(len(units)))
    axes[-1, 0].set_xticklabels([u.replace(" · ", "\n") for u in units], fontsize=9)
    handles = [Line2D([0], [0], marker="s", color="w", markerfacecolor="#e4e3de", markeredgecolor="#898781",
                      markersize=11, label="21 años sin Niño (rango completo)")]
    handles += [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markeredgecolor="white",
                       markersize=8, label=MAGNITUDE_LABELS[m].replace("\n", " ")) for m, c in mag_color.items()]
    axes[0, 0].legend(handles=handles, loc="upper left", bbox_to_anchor=(0, 1.32), ncol=4, frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.text(0.01, 0.005, "z = (media regional del año − media de referencia) / desviación estándar de las anomalías de los "
             "años sin Niño (cada uno contra los demás). Años etiquetados: fuera del rango de los años sin Niño.",
             fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.08, hspace=0.12)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------


def write_composite_tifs(pts, comps_by_var, grid, out_dir, variety):
    """GeoTIFFs of composite z and significance (1/0) for GIS users."""
    import rasterio

    written = []
    for v, comps in comps_by_var.items():
        for g, c in comps.items():
            for name, vals, dtype, nod in (("z", c["z"], "float32", -9999.0),
                                           ("sig", c["sig"].astype(float), "int16", -9999)):
                img = _grid_image(grid, pts["ID"].to_numpy(), vals)
                arr = np.where(np.isnan(img), nod, img).astype(dtype)
                p = Path(out_dir) / f"composite_{name}_{VARIABLES[v][0]}_{g.replace('+', '_')}_{variety}.tif"
                with rasterio.open(p, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                                   dtype=dtype, crs=grid["crs"], transform=grid["transform"], nodata=nod) as d:
                    d.write(arr, 1)
                    d.update_tags(variable=VARIABLES[v][0], group=g, years=",".join(map(str, c["years"])),
                                  quantity="mean standardised anomaly" if name == "z" else
                                  "1 = significant (permutation, BH-FDR 0.10)")
                written.append(p)
    return written


def run_el_nino_regional(
    *,
    results_root,
    potato_grid_file,
    variety: str,
    variety_label: str,
    events_csv,
    years=range(1981, 2020),
    min_overlap_months: int = 3,
    n_perm: int = 20_000,
    alpha_fdr: float = 0.10,
    gadm_peru_gpkg=None,
) -> dict:
    from load_variables import _raster_grid

    years = list(years)
    out_dir = Path(os.fspath(results_root)) / "el_nino_regional"
    out_dir.mkdir(parents=True, exist_ok=True)

    events = load_events(events_csv)
    yt = classify_years(events, years, min_overlap_months=min_overlap_months).set_index("year")
    ref_years = [y for y in years if not yt.loc[y, "is_nino"]]
    groups_years = {
        g: [y for y in years if yt.loc[y, "is_nino"] and yt.loc[y, "magnitude"] in mags]
        for g, mags in GROUPS.items()
    }

    pts, mats = load_matrix(Path(os.fspath(results_root)) / "yearly_aggregated_data.parquet", years)
    if any(np.isnan(m).any() for m in mats.values()):
        raise ValueError("missing point-years in the yearly table")

    tables, comps_by_var = [], {}
    for v, (short, _, _, sd_floor) in VARIABLES.items():
        comps_by_var[v] = composites(mats[v], years, groups_years, ref_years,
                                     sd_floor=sd_floor, n_perm=n_perm, alpha_fdr=alpha_fdr)
        tables.append(regional_table(pts, mats[v], years, yt, ref_years, var=v))
        print(f"[el_nino_regional] {variety} {short}: " + ", ".join(
            f"{g}: local p<0.05 {100 * c['share_local']:.1f}% (null ~{100 * c['field_null_median']:.1f}%), "
            f"field p={c['field_p']:.4f}, FDR {100 * c['sig_fdr'].mean():.2f}%"
            for g, c in comps_by_var[v].items()))
    table = pd.concat(tables, ignore_index=True)
    table.insert(0, "variety", variety)
    csv_path = out_dir / f"regional_el_nino_{variety}.csv"
    table.round(5).to_csv(csv_path, index=False)

    # per-point composite table (for further analysis)
    point_rows = pts[["ID", "lon", "lat", "region", "zone"]].copy()
    for v, comps in comps_by_var.items():
        for g, c in comps.items():
            k = f"{VARIABLES[v][0]}_{g.replace('+', '_')}"
            point_rows[f"z_{k}"] = c["z"].astype(np.float32)
            point_rows[f"delta_{k}"] = c["delta"].astype(np.float32)
            point_rows[f"p_{k}"] = c["p"].astype(np.float32)
            point_rows[f"sig_local_{k}"] = c["sig"]
            point_rows[f"sig_fdr_{k}"] = c["sig_fdr"]
    point_rows.to_parquet(out_dir / f"point_composites_{variety}.parquet", index=False)

    grid = _raster_grid(os.fspath(potato_grid_file))
    write_composite_tifs(pts, comps_by_var, grid, out_dir, variety)
    plot_composites(
        pts, comps_by_var, grid,
        title=f"Compuestos El Niño costero − años sin Niño, variedad {variety_label} ({years[0]}–{years[-1]}, "
              f"referencia: {len(ref_years)} años sin Niño)",
        out_path=out_dir / f"composites_el_nino_{variety}.png",
        gadm_peru_gpkg=gadm_peru_gpkg,
    )
    plot_dotbox(
        table,
        title=f"Anomalía regional por año El Niño frente a la variación de los años sin Niño, variedad {variety_label}",
        out_path=out_dir / f"regional_dotbox_el_nino_{variety}.png",
    )
    meta = {
        "reference_years": ref_years,
        "groups": groups_years,
        "weak_or_other_excluded": [y for y in years if yt.loc[y, "is_nino"]
                                   and y not in sum(groups_years.values(), [])],
        "n_perm": n_perm,
        "alpha_fdr": alpha_fdr,
        "field_significance": {
            VARIABLES[v][0]: {g: {"share_local_p05": c["share_local"], "null_median_share": c["field_null_median"],
                                  "field_p": c["field_p"], "share_fdr": float(c["sig_fdr"].mean())}
                              for g, c in comps.items()}
            for v, comps in comps_by_var.items()
        },
        "zones": ZONES,
    }
    import json

    with open(out_dir / f"metadata_{variety}.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"[el_nino_regional] {variety}: -> {out_dir}")
    return {"csv": os.fspath(csv_path), "dir": os.fspath(out_dir)}
