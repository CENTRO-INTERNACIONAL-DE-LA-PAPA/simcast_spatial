"""
Compare two SIMCAST result trees season by season (new run vs. previous run).

    python scripts/compare_results.py OLD_ROOT NEW_ROOT [--years 1981 2016]

For every `<year>/<period>/<region>/simcast_results.parquet` present in both
trees it reports, on the IDs/dates they share: number of IDs, date coverage,
max |diff| of HHR and temperature columns, and how many (ID, season) pairs
end with a different number of applications (APP).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["hr_ge_thresh", "tmean_ge_thresh", "rain_mm", "tavg_C", "BU", "FU", "APP"]


def compare_season(old_p: Path, new_p: Path, ids: np.ndarray | None = None) -> dict:
    old = pd.read_parquet(old_p)
    new = pd.read_parquet(new_p)
    if ids is not None:
        old = old[old.ID.isin(ids)]
        new = new[new.ID.isin(ids)]
    m = old.merge(new, on=["ID", "date"], suffixes=("_o", "_n"))
    out = {
        "ids_old": old.ID.nunique(),
        "ids_new": new.ID.nunique(),
        "ids_common": m.ID.nunique(),
        "dates_old": f"{old.date.min().date()}..{old.date.max().date()}",
        "dates_new": f"{new.date.min().date()}..{new.date.max().date()}",
        "rows_common": len(m),
    }
    for c in COLS:
        if f"{c}_o" in m:
            d = (m[f"{c}_o"].astype(float) - m[f"{c}_n"].astype(float)).abs()
            out[f"max|d {c}|"] = float(np.nanmax(d)) if len(d) else np.nan
    app_o = m.groupby("ID")["APP_o"].max()
    app_n = m.groupby("ID")["APP_n"].max()
    out["ids_app_differs"] = int((app_o != app_n).sum())
    out["mean_app_old"] = float(app_o.mean())
    out["mean_app_new"] = float(app_n.mean())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("old_root")
    ap.add_argument("new_root")
    ap.add_argument("--years", nargs=2, type=int, default=None)
    ap.add_argument("--ids", default=None, help="parquet/csv with an ID column to restrict")
    a = ap.parse_args(argv)

    ids = None
    if a.ids:
        p = Path(a.ids)
        ids = (pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p))["ID"].to_numpy()

    new_root = Path(a.new_root)
    rows = []
    for new_p in sorted(new_root.glob("*/*/*/simcast_results.parquet")):
        year, period, region = new_p.parts[-4:-1]
        if a.years and not (a.years[0] <= int(year) <= a.years[1]):
            continue
        old_p = Path(a.old_root) / year / period / region / "simcast_results.parquet"
        if not old_p.exists():
            continue
        r = compare_season(old_p, new_p, ids)
        r.update({"year": int(year), "period": period, "region": region})
        rows.append(r)
        print(f"{year} {period} {region}: " + ", ".join(f"{k}={v}" for k, v in r.items() if k not in ("year", "period", "region")))
    if rows:
        df = pd.DataFrame(rows)
        pd.set_option("display.width", 250)
        print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
