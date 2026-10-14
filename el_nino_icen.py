"""
How SIMCAST outputs respond to El Niño Costero, season by season.

Each potato point has ONE season per SIMCAST year, fixed by its region:

    Sierra Media               (cultivo3)  Apr - Jul of Y
    Costa y Valle Interandino  (cultivo2)  Jul - Dec of Y
    Sierra Alta                (cultivo1)  Nov of Y - Apr of Y+1

El Niño is measured during THAT season: the mean of the official monthly
ICEN (ENFEN 2024; ``reference/icen_enfen2024.txt``) over the season's months.
So a Sierra Alta season Nov 1998 - Apr 1999 (ICEN -0.37) is a neutral season
even though 1998 belongs to the 1997-98 event.

Analysis (1981-2019; 2020 has no Sierra Alta season)
----------------------------------------------------
* Per point: least-squares slope of the season value (APP, BU, FU) on the
  season-mean ICEN, over the 39 seasons (El Niño, La Niña and neutral).
  Mapped in standard deviations of the variable per +1 ICEN.
* Per unit (SIMCAST region x Norte >= -9 deg / Centro / Sur < -15 deg): the
  same regression on the unit-mean series, in variable units and %.
* Significance: permutation test that reassigns the ICEN series to the
  years in blocks of two consecutive years (random block offset), so a
  two-year event such as 1997-98 moves as one piece.  The same permutation
  is applied to every point, which keeps spatial correlation, and gives the
  field significance (Livezey & Chen 1983): is the share of points with
  local p < 0.05 larger than under the permutations?

Outputs in ``<results_root>/el_nino_icen/``
-------------------------------------------
* ``icen_slope_maps_<variety>.png``  - slope maps APP / BU / FU, stippled
  where local p < 0.05, field significance in each panel.
* ``icen_scatter_regions_<variety>.png`` - per unit and variable, season
  ICEN vs. unit anomaly with the fitted line, slope, r and p.
* ``icen_event_maps_<variety>.png`` - the mean of the neutral seasons, then
  one anomaly map per El Niño event; each region shows the season of that
  event with the highest ICEN (only if it was warm, ICEN > 0.5).
* ``icen_story_1991_vs_1997_<variety>.png`` - infographic: same warm coast,
  opposite effects in the highlands (rain -> humid days -> BU -> FU -> APP).
* ``icen_regional_<variety>.csv`` - the data behind the scatter plots.
* ``icen_summary_<variety>.json`` - regional slopes and field significance.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

SEASON_MONTHS = {  # region -> (first, last) month offset from January of Y
    "Sierra Media": (3, 6),
    "Costa y Valle Interandino": (6, 11),
    "Sierra Alta": (10, 15),
}
VARIABLES = {  # yearly-table column -> (label, anomaly unit for the scatter, sd floor)
    "APP": ("APP", "aplicaciones", 0.5),
    "ABU": ("BU", "%", 0.5),
    "AFU": ("FU", "%", 0.5),
}
ZONES = [("Norte", -9.0, 90.0), ("Centro", -15.0, -9.0), ("Sur", -90.0, -15.0)]
REGION_ORDER = ["Costa y Valle Interandino", "Sierra Media", "Sierra Alta"]
REGION_SHORT = {"Costa y Valle Interandino": "Costa/VI", "Sierra Media": "S. Media", "Sierra Alta": "S. Alta"}
REGION_SEASON = {"Costa y Valle Interandino": "jul–dic", "Sierra Media": "abr–jul", "Sierra Alta": "nov–abr"}

# ENFEN (2024) ICEN categories, applied to the season mean
CATEGORIES = [
    ("fría", -np.inf, -0.7),
    ("neutra", -0.7, 0.5),
    ("cálida débil", 0.5, 1.3),
    ("cálida moderada", 1.3, 2.1),
    ("cálida fuerte", 2.1, 3.5),
    ("cálida extraordinaria", 3.5, np.inf),
]


# ----------------------------------------------------------------------------
# ICEN
# ----------------------------------------------------------------------------


def load_icen(path) -> pd.Series:
    t = pd.read_csv(path, comment="%", sep=r"\s+", names=["year", "month", "icen"])
    idx = pd.PeriodIndex.from_fields(year=t["year"], month=t["month"], freq="M")
    return pd.Series(t["icen"].to_numpy(), index=idx).sort_index()


def season_icen(icen: pd.Series, years) -> pd.DataFrame:
    """Mean ICEN over each region's season months; index = year, columns = region."""
    out = {}
    for region, (a, b) in SEASON_MONTHS.items():
        vals = []
        for y in years:
            jan = pd.Period(f"{y}-01", freq="M")
            months = pd.period_range(jan + a, jan + b, freq="M")
            missing = [m for m in months if m not in icen.index]
            if missing:
                raise ValueError(f"ICEN missing for {missing[:3]}")
            vals.append(float(icen.loc[months].mean()))
        out[region] = vals
    return pd.DataFrame(out, index=list(years))


def category(v: float) -> str:
    for name, lo, hi in CATEGORIES:
        if lo <= v < hi:
            return name
    return "neutra"


def season_label(region: str, year: int) -> str:
    return f"{year}/{str(year + 1)[-2:]}" if region == "Sierra Alta" else str(year)


# ----------------------------------------------------------------------------
# Permutations
# ----------------------------------------------------------------------------


def block_permutations(n: int, n_perm: int, *, block: int = 2, seed: int = 0) -> np.ndarray:
    """(n_perm, n) permutations of 0..n-1 that move consecutive blocks together."""
    rng = np.random.default_rng(seed)
    perms = np.empty((n_perm, n), dtype=np.int64)
    for j in range(n_perm):
        off = rng.integers(0, block)
        cuts = [0] + list(range(off if off else block, n, block)) + [n]
        blocks = [np.arange(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]
        order = rng.permutation(len(blocks))
        perms[j] = np.concatenate([blocks[i] for i in order])
    return perms


# ----------------------------------------------------------------------------
# Regression
# ----------------------------------------------------------------------------


def slopes_with_permutation(y: np.ndarray, x: np.ndarray, perms: np.ndarray, *,
                            local_alpha: float = 0.05, chunk: int = 2_000):
    """
    OLS slope of each row of ``y`` (points x years) on ``x`` (years), with
    permutation p-values and the per-permutation count of locally
    significant rows (for field significance).
    """
    xc = x - x.mean()
    ssx = float(xc @ xc)
    xp = xc[perms].T.astype(np.float32)  # (years, n_perm)
    yc = (y - y.mean(axis=1, keepdims=True)).astype(np.float32)
    b = (yc @ xc.astype(np.float32)) / ssx
    p = np.empty(len(y))
    field = np.zeros(perms.shape[0], dtype=np.int64)
    for a in range(0, len(y), chunk):
        bp = np.abs((yc[a:a + chunk] @ xp) / ssx)
        obs = np.abs(b[a:a + chunk])[:, None]
        p[a:a + chunk] = ((bp >= obs - 1e-9).sum(axis=1) + 1) / (perms.shape[0] + 1)
        q = np.quantile(bp, 1 - local_alpha, axis=1, method="higher")[:, None]
        flat = bp.max(axis=1, keepdims=True) <= 1e-9
        field += ((bp >= q - 1e-9) & ~flat).sum(axis=0)
    return b, p, field


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------


def zone_of(lat: np.ndarray) -> np.ndarray:
    out = np.empty(len(lat), dtype=object)
    for name, lo, hi in ZONES:
        out[(lat >= lo) & (lat < hi)] = name
    return out


def load_matrix(yearly_path, years):
    y = pd.read_parquet(yearly_path, columns=["ID", "lon", "lat", "region", "year", *VARIABLES])
    y = y[y["year"].isin(years)]
    pts = y.drop_duplicates("ID")[["ID", "lon", "lat", "region"]].sort_values("ID").reset_index(drop=True)
    pts["zone"] = zone_of(pts["lat"].to_numpy())
    pts["unit"] = pts["region"].map(REGION_SHORT) + " · " + pts["zone"]
    r = pd.Index(pts["ID"]).get_indexer(y["ID"])
    c = pd.Index(list(years)).get_indexer(y["year"])
    mats = {}
    for v in VARIABLES:
        m = np.full((len(pts), len(years)), np.nan)
        m[r, c] = y[v].to_numpy(dtype=np.float64)
        if np.isnan(m).any():
            raise ValueError(f"missing point-years for {v}")
        mats[v] = m
    return pts, mats


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------


def plot_slope_maps(pts, maps, grid, *, title, out_path, gadm_peru_gpkg=None):
    import matplotlib.pyplot as plt

    from viz_palette import diverging_classes

    classes = diverging_classes(np.concatenate([m["slope_sd"] for m in maps.values()]), n_per_arm=4)
    cmap, norm = classes.cmap_norm()
    adm0 = adm1 = None
    if gadm_peru_gpkg is not None:
        try:
            from animation_all_lands import load_gadm_peru

            adm0, adm1 = load_gadm_peru(gadm_peru_gpkg)
        except Exception as e:
            print(f"[el_nino_icen] no GADM: {e}")

    t = grid["transform"]
    extent = [t.c, t.c + grid["ncols"] * t.a, t.f + grid["nrows"] * t.e, t.f]
    pos = pd.Index(grid["ids"]).get_indexer(pts["ID"])
    yy, xx = grid["y_idx"][pos], grid["x_idx"][pos]
    fig, axes = plt.subplots(1, len(maps), figsize=(5.0 * len(maps), 7.2))
    im = None
    for ax, (v, m) in zip(np.atleast_1d(axes), maps.items()):
        img = np.full((grid["nrows"], grid["ncols"]), np.nan)
        img[yy, xx] = m["slope_sd"]
        im = ax.imshow(np.ma.masked_invalid(img), extent=extent, origin="upper", cmap=cmap, norm=norm,
                       interpolation="nearest")
        # stippling: 10x10-pixel blocks where most points have local p < 0.05
        blk = pd.DataFrame({"by": yy // 10, "bx": xx // 10, "s": m["p"] < 0.05}).groupby(["by", "bx"])["s"].mean()
        blk = blk[blk >= 0.5]
        ax.scatter(t.c + (blk.index.get_level_values(1) * 10 + 5) * t.a,
                   t.f + (blk.index.get_level_values(0) * 10 + 5) * t.e, s=0.7, c="#0b0b0b", linewidths=0)
        if adm1 is not None:
            adm1.boundary.plot(ax=ax, edgecolor="0.6", linewidth=0.3)
            adm0.boundary.plot(ax=ax, edgecolor="black", linewidth=0.6)
        for lat in (-9, -15):
            ax.axhline(lat, color="0.35", lw=0.6, ls=(0, (4, 3)))
        ax.set_xlim(-81.3, -68.5)
        ax.set_ylim(-18.5, -3.3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(VARIABLES[v][0], fontsize=13, fontweight="bold")
        fp = m["field_p"]
        ax.text(0.02, 0.02,
                f"p < 0.05 local: {100 * m['share_local']:.1f} % de puntos\n"
                f"(esperable sin relación: ~{100 * m['null_share']:.0f} %)\n"
                f"significancia de campo: p = {fp:.3f}" if fp >= 0.001 else
                f"p < 0.05 local: {100 * m['share_local']:.1f} % de puntos\n"
                f"(esperable sin relación: ~{100 * m['null_share']:.0f} %)\n"
                f"significancia de campo: p < 0.001",
                transform=ax.transAxes, fontsize=9, va="bottom")
    import textwrap

    fig.suptitle("\n".join(textwrap.wrap(title, 95)), fontsize=12)
    cax = fig.add_axes([0.2, 0.08, 0.6, 0.025])
    classes.colorbar(fig, im, cax, label="Cambio por +1 de ICEN medio de la campaña (desviaciones estándar de la variable)")
    fig.text(0.5, 0.01, "Puntos negros: p < 0.05 local. Permutación por bloques de 2 años (5000); significancia de campo "
             "según Livezey y Chen (1983). Líneas discontinuas: −9° y −15°.", ha="center", fontsize=8)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.17, wspace=0.03)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_scatter(table, summary, *, title, out_path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from viz_palette import BLUE, RED

    cat_color = {
        "fría": BLUE[400], "neutra": "#c3c2b7", "cálida débil": RED[200], "cálida moderada": RED[350],
        "cálida fuerte": RED[500], "cálida extraordinaria": RED[700],
    }
    units = [(r, z) for r in REGION_ORDER for z, _, _ in ZONES]
    vars_ = [VARIABLES[v][0] for v in VARIABLES]
    fig, axes = plt.subplots(len(vars_), len(units), figsize=(2.35 * len(units), 2.6 * len(vars_) + 1.4),
                             sharex="col", sharey="row", squeeze=False)
    for i, var in enumerate(vars_):
        unit_label = dict((VARIABLES[k][0], VARIABLES[k][1]) for k in VARIABLES)[var]
        col = "anomaly" if unit_label == "aplicaciones" else "anomaly_pct"
        for j, (region, zone) in enumerate(units):
            ax = axes[i, j]
            unit = f"{REGION_SHORT[region]} · {zone}"
            d = table[(table["unit"] == unit) & (table["variable"] == var)]
            ax.axhline(0, color="#e1e0d9", lw=0.8, zorder=0)
            ax.axvline(0.5, color="#e1e0d9", lw=0.8, ls=(0, (3, 2)), zorder=0)
            ax.scatter(d["icen_season"], d[col], s=16, c=d["category"].map(cat_color),
                       edgecolors="white", linewidths=0.6, zorder=3)
            s = summary["regional"][var][unit]
            xs = np.linspace(d["icen_season"].min(), d["icen_season"].max(), 2)
            slope = s["slope"] if col == "anomaly" else s["slope_pct"]
            ax.plot(xs, slope * (xs - d["icen_season"].mean()) + d[col].mean(), color="#52514e", lw=1.2, zorder=2)
            for _, r in d[d["icen_season"] > 2.1].iterrows():
                ax.annotate(r["season"], (r["icen_season"], r[col]), xytext=(3, 2), textcoords="offset points",
                            fontsize=6.5, color="#52514e")
            p = s["p_block"]
            ax.text(0.03, 0.97, f"{slope:+.2f}{'' if col == 'anomaly' else ' %'} por +1\nr = {s['r']:.2f}, "
                    f"p = {p:.3f}" if p >= 0.001 else f"{slope:+.2f}{'' if col == 'anomaly' else ' %'} por +1\n"
                    f"r = {s['r']:.2f}, p < 0.001",
                    transform=ax.transAxes, fontsize=7, va="top",
                    fontweight="bold" if p < 0.05 else "normal")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            lo, hi = d["icen_season"].min(), d["icen_season"].max()
            ax.set_xlim(lo - 0.3, hi + 0.25 * (hi - lo) + 0.3)  # room for the year labels
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(f"{REGION_SHORT[region]} ({REGION_SEASON[region]})\n{zone}", fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f"{var}: anomalía\n({unit_label})", fontsize=8.5)
            if i == len(vars_) - 1:
                ax.set_xlabel("ICEN medio de la campaña", fontsize=8)
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=7, label=k)
               for k, c in cat_color.items()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=6, frameon=False, fontsize=8,
               title="Condición durante la campaña (categorías ICEN, ENFEN 2024)", title_fontsize=8)
    fig.suptitle(title, fontsize=11, y=0.995)
    fig.text(0.01, 0.005, "Cada punto es una campaña 1981–2019; anomalía respecto a la media 1981–2019 de la unidad. "
             "Línea: ajuste lineal. p: permutación por bloques de 2 años. En negrita: p < 0.05. "
             "Etiquetas: campañas con ICEN medio > 2.1 (fuerte o extraordinaria).", fontsize=7)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.82, bottom=0.08, wspace=0.08, hspace=0.12)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


# El Niño Costero events (ENFEN 2024) with at least one moderate-or-stronger
# regional season, as groups of SIMCAST years.
EVENTS = {"1982–83": (1982, 1983), "1986–87": (1986, 1987), "1991–92": (1991, 1992),
          "1997–98": (1997, 1998), "2015–16": (2015, 2016)}


def event_seasons(x_by_region: pd.DataFrame, events=EVENTS, warm: float = 0.5) -> dict:
    """{event: {region: (year, icen) or None}}: the event's warmest season per region."""
    out = {}
    for name, ys in events.items():
        out[name] = {}
        for region in SEASON_MONTHS:
            cand = [(y, float(x_by_region.loc[y, region])) for y in ys if y in x_by_region.index]
            y, v = max(cand, key=lambda t: t[1])
            out[name][region] = (y, v) if v > warm else None
    return out


def plot_event_maps(pts, mats, x_by_region, years, grid, *, title, out_path, gadm_peru_gpkg=None):
    """Rows APP/BU/FU; neutral-season mean, then one anomaly map per event."""
    import matplotlib.pyplot as plt

    from viz_palette import diverging_classes, sequential_classes

    ev = event_seasons(x_by_region)
    yi = {y: i for i, y in enumerate(years)}
    regions = pts["region"].to_numpy()
    neutral = {r: [yi[y] for y in years if -0.7 <= x_by_region.loc[y, r] < 0.5] for r in SEASON_MONTHS}

    t = grid["transform"]
    extent = [t.c, t.c + grid["ncols"] * t.a, t.f + grid["nrows"] * t.e, t.f]
    pos = pd.Index(grid["ids"]).get_indexer(pts["ID"])
    yy, xx = grid["y_idx"][pos], grid["x_idx"][pos]

    def image(vals):
        img = np.full((grid["nrows"], grid["ncols"]), np.nan)
        img[yy, xx] = vals
        return np.ma.masked_invalid(img)

    adm0 = adm1 = None
    if gadm_peru_gpkg is not None:
        try:
            from animation_all_lands import load_gadm_peru

            adm0, adm1 = load_gadm_peru(gadm_peru_gpkg)
        except Exception as e:
            print(f"[el_nino_icen] no GADM: {e}")

    ncol = 1 + len(ev)
    fig, axes = plt.subplots(3, ncol, figsize=(2.6 * ncol + 1.0, 3 * 3.25 + 2.2), squeeze=False)
    fig.subplots_adjust(left=0.035, right=0.925, top=0.86, bottom=0.07, wspace=0.04, hspace=0.16)
    for i, (v, (label, unit, _)) in enumerate(VARIABLES.items()):
        x = mats[v]
        ref = np.empty(len(pts))
        for r, cols in neutral.items():
            m = regions == r
            ref[m] = x[m][:, cols].mean(axis=1)
        anoms = {}
        for name, per_region in ev.items():
            a = np.full(len(pts), np.nan)
            for r, chosen in per_region.items():
                if chosen is not None:
                    m = regions == r
                    a[m] = x[m, yi[chosen[0]]] - ref[m]
            anoms[name] = a
        seq = sequential_classes(ref)
        allan = np.concatenate([a[np.isfinite(a)] for a in anoms.values()])
        div = diverging_classes(allan, n_per_arm=4)
        scm, snm = seq.cmap_norm()
        dcm, dnm = div.cmap_norm()
        unit_txt = "aplicaciones" if unit == "aplicaciones" else label
        panels = [("Promedio campañas neutras", ref, None)] + [(n, anoms[n], ev[n]) for n in ev]
        im_s = im_d = None
        for j, (name, vals, info) in enumerate(panels):
            ax = axes[i, j]
            if info is None:
                im_s = ax.imshow(image(vals), extent=extent, origin="upper", cmap=scm, norm=snm,
                                 interpolation="nearest")
            else:
                im_d = ax.imshow(image(vals), extent=extent, origin="upper", cmap=dcm, norm=dnm,
                                 interpolation="nearest")
            if adm1 is not None:
                adm1.boundary.plot(ax=ax, edgecolor="0.6", linewidth=0.3)
                adm0.boundary.plot(ax=ax, edgecolor="black", linewidth=0.6)
            ax.set_xlim(-81.3, -68.5)
            ax.set_ylim(-18.5, -3.3)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                if info is None:
                    sub = (f"Costa/VI: {len(neutral['Costa y Valle Interandino'])} campañas\n"
                           f"S. Media: {len(neutral['Sierra Media'])} campañas\n"
                           f"S. Alta: {len(neutral['Sierra Alta'])} campañas")
                    head = "Promedio campañas neutras"
                else:
                    sub = "\n".join(
                        f"{REGION_SHORT[r]}: " + ("sin Niño" if info[r] is None else
                                                  f"{season_label(r, info[r][0])} (ICEN {info[r][1]:.1f})")
                        for r in REGION_ORDER)
                    head = f"El Niño {name}"
                ax.set_title(head, fontsize=10.5, fontweight="bold", pad=44)
                ax.text(0.5, 1.02, sub, transform=ax.transAxes, ha="center", va="bottom", fontsize=7.3,
                        color="#52514e")
            if j == 0:
                ax.set_ylabel(label, fontsize=13, fontweight="bold")
        # colourbars for the row: neutral mean (left, under the first map) and anomalies (right)
        b0 = axes[i, 0].get_position()
        cax = fig.add_axes([b0.x0 + 0.01, b0.y0 - 0.012, b0.width - 0.02, 0.007])
        seq.colorbar(fig, im_s, cax)
        cax.tick_params(labelsize=6.5)
        bl = axes[i, -1].get_position()
        cax2 = fig.add_axes([bl.x1 + 0.008, bl.y0 + 0.02, 0.008, bl.height - 0.04])
        cb = fig.colorbar(im_d, cax=cax2, extend="both", spacing="uniform", ticks=div.boundaries)
        cb.ax.set_yticklabels([f"{b:g}".replace("-", "−") for b in div.boundaries])
        cb.ax.tick_params(labelsize=7, length=0)
        cb.outline.set_visible(False)
        cb.set_label(f"Anomalía ({unit_txt} por campaña)", fontsize=8)
    import textwrap

    fig.suptitle("\n".join(textwrap.wrap(title, 110)), fontsize=12)
    fig.text(0.01, 0.004, "Anomalía = campaña del evento − promedio de las campañas neutras (ICEN medio entre −0.7 y 0.5) "
             "del mismo punto. En cada evento, cada región muestra su campaña con el ICEN más alto; en blanco si "
             "esa campaña no fue cálida (ICEN ≤ 0.5). Un solo evento por mapa: sin prueba de significancia.",
             fontsize=7.5)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Story figure: two El Niño events side by side
# ----------------------------------------------------------------------------

CULTIVO = {"Sierra Media": "cultivo3", "Costa y Valle Interandino": "cultivo2", "Sierra Alta": "cultivo1"}
WET_HOURS = 7  # fewest daily hours with RH >= 90 % that produce blight units


def _season_climate(args):
    """Per point of one season: total rain (mm) and days with >= WET_HOURS humid hours."""
    root, region, year = args
    f = Path(root) / str(year) / CULTIVO[region] / region / "simcast_results.parquet"
    d = pd.read_parquet(f, columns=["ID", "hr_ge_thresh", "rain_mm"])
    d["wet"] = d["hr_ge_thresh"] >= WET_HOURS
    g = d.groupby("ID").agg(rain=("rain_mm", "sum"), wet=("wet", "sum"))
    return region, year, g


def point_climate(results_root, seasons, workers: int = 8) -> dict:
    """{(region, year): DataFrame[ID -> rain, wet]} for the requested seasons."""
    import concurrent.futures as cf

    jobs = [(os.fspath(results_root), r, y) for r, y in sorted(set(seasons))]
    out = {}
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for region, year, g in ex.map(_season_climate, jobs):
            out[(region, year)] = g
    return out


def plot_event_story(pts, mats, x_by_region, years, grid, results_root, icen, *,
                     events=("1991–92", "1997–98"), variety_label="", out_path, gadm_peru_gpkg=None):
    """
    Infographic: the same warm coastal index, opposite effects in the
    highlands.  Columns = events; rows = rain -> humid days -> BU -> FU -> APP.
    Each region uses its own season of the event (warmest ICEN).
    """
    import textwrap

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    from viz_palette import NEUTRAL, RED, diverging_classes

    ev_all = event_seasons(x_by_region)
    ev = {e: ev_all[e] for e in events}
    yi = {y: i for i, y in enumerate(years)}
    regions = pts["region"].to_numpy()
    zones = pts["zone"].to_numpy()
    ids = pts["ID"].to_numpy()
    neutral = {r: [y for y in years if -0.7 <= x_by_region.loc[y, r] < 0.5] for r in SEASON_MONTHS}

    need = [(r, y) for r, ys in neutral.items() for y in ys]
    need += [(r, c[0]) for per in ev.values() for r, c in per.items() if c is not None]
    clim = point_climate(results_root, need)

    def clim_array(region, ys, col):
        stack = np.stack([clim[(region, y)][col].reindex(ids[regions == region]).to_numpy() for y in ys])
        return stack.mean(axis=0)

    rows = [
        ("rain", "Lluvia en la campaña", "% respecto a lo normal"),
        ("wet", f"Días con ≥ {WET_HOURS} h de humedad ≥ 90 %", "días respecto a lo normal"),
        ("ABU", "Presión de tizón (BU)", "BU respecto a lo normal"),
        ("AFU", "Lavado del fungicida (FU)", "FU respecto a lo normal"),
        ("APP", "Aplicaciones de fungicida", "aplicaciones respecto a lo normal"),
    ]
    anom = {k: {} for k, _, _ in rows}
    ref_rain = np.full(len(pts), np.nan)
    for r in SEASON_MONTHS:
        ref_rain[regions == r] = clim_array(r, neutral[r], "rain")
    for e, per in ev.items():
        for k, _, _ in rows:
            anom[k][e] = np.full(len(pts), np.nan)
        rain_e = np.full(len(pts), np.nan)
        for r, c in per.items():
            if c is None:
                continue
            m = regions == r
            y = c[0]
            rain_e[m] = clim[(r, y)]["rain"].reindex(ids[m]).to_numpy()
            anom["wet"][e][m] = clim[(r, y)]["wet"].reindex(ids[m]).to_numpy() - clim_array(r, neutral[r], "wet")
            for k in ("ABU", "AFU", "APP"):
                ref = mats[k][m][:, [yi[q] for q in neutral[r]]].mean(axis=1)
                anom[k][e][m] = mats[k][m, yi[y]] - ref
        with np.errstate(divide="ignore", invalid="ignore"):
            anom["rain"][e] = np.where(ref_rain >= 5, 100 * (rain_e / ref_rain - 1), np.nan)
        anom["rain"][e + "_abs"] = rain_e

    def zone_text(k, e):
        out = []
        for z in ("Norte", "Sur"):
            m = zones == z
            if k == "rain":
                tot_e = np.nansum(anom["rain"][e + "_abs"][m])
                tot_r = np.nansum(ref_rain[m])
                v = 100 * (tot_e / tot_r - 1)
                out.append(f"{z}: {v:+.0f} %")
            else:
                v = np.nanmean(anom[k][e][m])
                fmt = "{:+.1f}" if k in ("APP", "wet") else "{:+.0f}"
                out.append(f"{z}: " + fmt.format(v))
        return "   ".join(out)

    # ---- layout --------------------------------------------------------
    t = grid["transform"]
    extent = [t.c, t.c + grid["ncols"] * t.a, t.f + grid["nrows"] * t.e, t.f]
    pos = pd.Index(grid["ids"]).get_indexer(ids)
    yy, xx = grid["y_idx"][pos], grid["x_idx"][pos]

    def image(vals):
        img = np.full((grid["nrows"], grid["ncols"]), np.nan)
        img[yy, xx] = vals
        return np.ma.masked_invalid(img)

    adm0 = adm1 = None
    if gadm_peru_gpkg is not None:
        try:
            from animation_all_lands import load_gadm_peru

            adm0, adm1 = load_gadm_peru(gadm_peru_gpkg)
        except Exception as e_:
            print(f"[el_nino_icen] no GADM: {e_}")

    fig = plt.figure(figsize=(11.5, 30))
    gs = GridSpec(1 + len(rows), 3, figure=fig, width_ratios=[1, 1, 0.07],
                  height_ratios=[0.55] + [1] * len(rows), hspace=0.32, wspace=0.05,
                  left=0.13, right=0.92, top=0.915, bottom=0.075)

    fig.text(0.5, 0.985, "Dos Niños costeros, dos efectos opuestos en la sierra",
             ha="center", va="top", fontsize=19, fontweight="bold")
    fig.text(0.5, 0.968, "\n".join(textwrap.wrap(
        "Ambos eventos calentaron el mar frente a la costa (ICEN > 0.5). Lo que decide el tizón tardío es lo que "
        "pasa en la sierra durante la campaña: la lluvia y las horas de alta humedad. "
        f"Variedad {variety_label}. Rojo: más que en una campaña normal; azul: menos.", 105)),
             ha="center", va="top", fontsize=10.5, color="#52514e")

    # row 0: ICEN time series with the seasons used
    season_color = {"Costa y Valle Interandino": "#898781", "Sierra Media": "#52514e", "Sierra Alta": "#0b0b0b"}
    for j, (e, per) in enumerate(ev.items()):
        ax = fig.add_subplot(gs[0, j])
        y0, y1 = EVENTS[e]
        span = pd.period_range(f"{y0}-01", f"{y1 + 1}-06", freq="M")
        vals = icen.reindex(span).to_numpy()
        xs = np.arange(len(span))
        ax.axhspan(0.5, 6, color=RED[100], alpha=0.5, lw=0)
        ax.axhline(0.5, color=RED[300], lw=0.8, ls=(0, (3, 2)))
        ax.plot(xs, vals, color="#0b0b0b", lw=2)
        for k, r in enumerate(REGION_ORDER):
            c = per[r]
            if c is None:
                continue
            a, b = SEASON_MONTHS[r]
            start = (c[0] - y0) * 12 + a
            ax.plot([start, start + (b - a)], [-1.25 - 0.32 * k] * 2, color=season_color[r], lw=5,
                    solid_capstyle="butt")
            ax.text(start + (b - a) + 0.6, -1.25 - 0.32 * k, f"{REGION_SHORT[r]} (ICEN {c[1]:.1f})",
                    va="center", fontsize=7.5, color=season_color[r])
        ax.set_ylim(-2.2, 4.6)
        ax.set_xlim(0, len(span) + 9)
        ticks = [i for i, pnt in enumerate(span) if pnt.month == 1]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(span[i].year) for i in ticks], fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title(f"El Niño {e}", fontsize=15, fontweight="bold", pad=8)
        if j == 0:
            ax.set_ylabel("ICEN\n(mar costero)", fontsize=9)
        ax.text(0.01, 0.97, "zona cálida", transform=ax.transAxes, fontsize=7, color=RED[450], va="top")

    # rows 1..: maps
    for i, (k, name, unit) in enumerate(rows, start=1):
        vals_all = np.concatenate([anom[k][e][np.isfinite(anom[k][e])] for e in ev])
        cls = diverging_classes(vals_all, n_per_arm=4, percentile=97)
        cmap, norm = cls.cmap_norm()
        im = None
        for j, e in enumerate(ev):
            ax = fig.add_subplot(gs[i, j])
            im = ax.imshow(image(anom[k][e]), extent=extent, origin="upper", cmap=cmap, norm=norm,
                           interpolation="nearest")
            if adm1 is not None:
                adm1.boundary.plot(ax=ax, edgecolor="0.62", linewidth=0.3)
                adm0.boundary.plot(ax=ax, edgecolor="black", linewidth=0.6)
            ax.axhline(-9, color="0.45", lw=0.5, ls=(0, (4, 3)))
            ax.axhline(-15, color="0.45", lw=0.5, ls=(0, (4, 3)))
            ax.set_xlim(-81.3, -68.5)
            ax.set_ylim(-18.5, -3.3)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, -0.025, zone_text(k, e), transform=ax.transAxes, ha="center", va="top",
                    fontsize=10, fontweight="bold", color="#0b0b0b")
            if j == 0:
                ax.text(-0.06, 0.5, f"{i}. {name}", transform=ax.transAxes, rotation=90, ha="right",
                        va="center", fontsize=12, fontweight="bold")
        cax = fig.add_subplot(gs[i, 2])
        cb = fig.colorbar(im, cax=cax, extend="both", spacing="uniform", ticks=cls.boundaries)
        cb.ax.set_yticklabels([f"{b:g}".replace("-", "−") for b in cls.boundaries])
        cb.ax.tick_params(labelsize=8, length=0)
        cb.outline.set_visible(False)
        cb.set_label(unit, fontsize=8.5)
        if i < len(rows):
            fig.text(0.075, cax.get_position().y0 - 0.006, "↓", ha="center", va="center", fontsize=20,
                     color="#898781")

    takeaway = [
        ("1991–92, Niño seco en la sierra:", "menos lluvia → menos lavado del fungicida y, en el sur, menos horas "
         "húmedas → menos tizón → menos aplicaciones."),
        ("1997–98, Niño húmedo en el norte:", "más lluvia y más días húmedos → más tizón y más lavado → más "
         "aplicaciones. En el sur, menos horas húmedas y menos lavado → menos aplicaciones."),
        ("Conclusión:", "el índice del mar no basta para anticipar el tizón; hay que mirar el pronóstico de lluvia "
         "y humedad en la sierra para la campaña."),
    ]
    txt = "\n".join("\n".join(textwrap.wrap(f"{a} {b}", 118, subsequent_indent="    ")) for a, b in takeaway)
    fig.text(0.07, 0.058, txt, ha="left", va="top", fontsize=10.3, linespacing=1.35,
             bbox=dict(boxstyle="round,pad=0.7", facecolor=NEUTRAL, edgecolor="none"))
    note = ("Normal = promedio de las campañas neutras (ICEN −0.7 a 0.5) del mismo punto. Cada región usa su campaña "
            "del evento con el ICEN más alto. Cifras bajo los mapas: promedio de la zona norte (> −9°) y sur "
            "(< −15°); líneas discontinuas: −9° y −15°. Fuente: SIMCAST con PISCOt v1.2, PISCOeo_pm/TimesFM y "
            "PISCOp v3.0; ICEN: ENFEN 2024.")
    fig.text(0.07, 0.012, "\n".join(textwrap.wrap(note, 175)), fontsize=7, color="#52514e", va="bottom")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------


def run_el_nino_icen(
    *,
    results_root,
    potato_grid_file,
    icen_file,
    variety: str,
    variety_label: str,
    years=range(1981, 2020),
    n_perm: int = 5000,
    gadm_peru_gpkg=None,
) -> dict:
    from load_variables import _raster_grid

    years = list(years)
    out_dir = Path(os.fspath(results_root)) / "el_nino_icen"
    out_dir.mkdir(parents=True, exist_ok=True)

    icen = load_icen(icen_file)
    x_by_region = season_icen(icen, years)
    pts, mats = load_matrix(Path(os.fspath(results_root)) / "yearly_aggregated_data.parquet", years)
    perms = block_permutations(len(years), n_perm)

    maps, rows = {}, []
    summary = {"years": [years[0], years[-1]], "n_perm": n_perm, "block_years": 2,
               "field": {}, "regional": {}}
    for v, (label, _, sd_floor) in VARIABLES.items():
        y = mats[v]
        b = np.zeros(len(pts))
        p = np.ones(len(pts))
        field = np.zeros(n_perm)
        for region in SEASON_MONTHS:
            idx = np.flatnonzero(pts["region"].to_numpy() == region)
            br, pr, fr = slopes_with_permutation(y[idx], x_by_region[region].to_numpy(), perms)
            b[idx], p[idx] = br, pr
            field += fr
        constant = y.std(axis=1) == 0
        p[constant] = 1.0
        sd = np.maximum(y.std(axis=1, ddof=1), sd_floor)
        share = float(np.mean(p < 0.05))
        null_share = field / len(pts)
        maps[v] = {"slope_sd": b / sd, "p": p, "share_local": share,
                   "null_share": float(np.median(null_share)),
                   "field_p": float(np.mean(null_share >= share - 1e-12))}
        summary["field"][label] = {k: maps[v][k] for k in ("share_local", "null_share", "field_p")}

        # regional series
        summary["regional"][label] = {}
        for unit, g in pts.groupby("unit", sort=False):
            region = g["region"].iloc[0]
            series = y[g.index.to_numpy()].mean(axis=0)
            mean = series.mean()
            xr = x_by_region[region].to_numpy()
            bu, pu, _ = slopes_with_permutation(series[None, :], xr, perms)
            r = float(np.corrcoef(xr, series)[0, 1]) if series.std() > 0 else 0.0
            summary["regional"][label][unit] = {
                "slope": float(bu[0]), "slope_pct": float(100 * bu[0] / mean) if mean else None,
                "r": r, "p_block": float(pu[0]), "n_points": int(len(g)), "mean": float(mean),
            }
            for k, yr in enumerate(years):
                rows.append({
                    "variety": variety, "variable": label, "unit": unit, "region": region,
                    "zone": g["zone"].iloc[0], "year": yr, "season": season_label(region, yr),
                    "icen_season": xr[k], "category": category(xr[k]), "unit_mean": series[k],
                    "anomaly": series[k] - mean, "anomaly_pct": 100 * (series[k] - mean) / mean if mean else np.nan,
                    "n_points": len(g),
                })
        print(f"[el_nino_icen] {variety} {label}: local p<0.05 {100 * share:.1f}% "
              f"(null ~{100 * maps[v]['null_share']:.1f}%), field p = {maps[v]['field_p']:.4f}")

    table = pd.DataFrame(rows)
    csv_path = out_dir / f"icen_regional_{variety}.csv"
    table.round(5).to_csv(csv_path, index=False)
    with open(out_dir / f"icen_summary_{variety}.json", "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    grid = _raster_grid(os.fspath(potato_grid_file))
    plot_slope_maps(pts, maps, grid,
                    title=f"Respuesta a El Niño costero durante la campaña, variedad {variety_label}: "
                          f"pendiente de cada variable frente al ICEN medio de su campaña ({years[0]}–{years[-1]})",
                    out_path=out_dir / f"icen_slope_maps_{variety}.png", gadm_peru_gpkg=gadm_peru_gpkg)
    plot_event_maps(pts, mats, x_by_region, years, grid,
                    title=f"Promedio de las campañas neutras y anomalía en cada evento El Niño costero, "
                          f"variedad {variety_label}",
                    out_path=out_dir / f"icen_event_maps_{variety}.png", gadm_peru_gpkg=gadm_peru_gpkg)
    summary["event_seasons"] = {
        n: {r: (None if c is None else {"season": season_label(r, c[0]), "icen": round(c[1], 2)})
            for r, c in per.items()} for n, per in event_seasons(x_by_region).items()}
    with open(out_dir / f"icen_summary_{variety}.json", "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    plot_event_story(pts, mats, x_by_region, years, grid, results_root, icen, variety_label=variety_label,
                     out_path=out_dir / f"icen_story_1991_vs_1997_{variety}.png", gadm_peru_gpkg=gadm_peru_gpkg)
    plot_scatter(table, summary,
                 title=f"ICEN medio de cada campaña frente a la anomalía regional, variedad {variety_label}",
                 out_path=out_dir / f"icen_scatter_regions_{variety}.png")
    return {"dir": os.fspath(out_dir), "csv": os.fspath(csv_path)}
