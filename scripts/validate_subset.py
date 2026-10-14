"""
Re-run SIMCAST for a sample of points of one season and compare with a
previous result tree.

    python scripts/validate_subset.py --year 2015 --period cultivo3 \
        --region "Sierra Media" --n 2000 --sun-method astral --timezone America/Bogota

Prints the max absolute differences per column and the number of points whose
final number of applications differs.  Run from the repo root.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import EXAMPLE_PATHS  # noqa: E402
from load_variables import (  # noqa: E402
    process_region_period,
    season_periods,
    split_regions_by_type,
)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2015)
    ap.add_argument("--period", default="cultivo3")
    ap.add_argument("--region", default="Sierra Media")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sun-method", default="astral")
    ap.add_argument("--timezone", default="America/Bogota")
    ap.add_argument("--vt", default="r")
    ap.add_argument("--old-root", default="/media/ppalacios/Data1/henry_simcast_peru/results_resistant")
    ap.add_argument("--base-path", default=None)
    a = ap.parse_args(argv)

    import geopandas as gpd

    base_path = a.base_path or EXAMPLE_PATHS.parquet_base_dir
    grid = gpd.read_file(EXAMPLE_PATHS.potato_grid_file)
    if "ID" not in grid.columns:
        grid["ID"] = grid.index
    region = split_regions_by_type(grid)[a.region]
    rng = np.random.default_rng(a.seed)
    sample = region.iloc[np.sort(rng.choice(len(region), size=min(a.n, len(region)), replace=False))]
    dr = season_periods(a.year)[a.period]

    t0 = time.perf_counter()
    new = process_region_period(
        sample,
        dr,
        a.vt,
        base_path=base_path,
        timezone=a.timezone,
        sun_method=a.sun_method,
    )
    dt = time.perf_counter() - t0
    print(f"new run: {len(sample)} IDs, {len(new)} rows, {dt:.1f}s "
          f"({dt / max(1, len(sample)) * 1000:.1f} ms/ID incl. I/O)")
    if new.empty:
        print("no results")
        return 1
    print("new dates:", new.date.min().date(), "..", new.date.max().date())

    old_p = Path(a.old_root) / str(a.year) / a.period / a.region / "simcast_results.parquet"
    if not old_p.exists():
        print("no previous results at", old_p)
        return 0
    old = pd.read_parquet(old_p, filters=[("ID", "in", sample["ID"].tolist())])
    print("old dates:", old.date.min().date(), "..", old.date.max().date(), "ids", old.ID.nunique())

    m = old.merge(new, on=["ID", "date"], suffixes=("_o", "_n"))
    print(f"common rows {len(m)} (old {len(old)}, new {len(new)})")
    for c in ["rain_mm", "tavg_C", "hr_ge_thresh", "tmean_ge_thresh", "BU", "FU", "APP"]:
        d = (m[f"{c}_o"].astype(float) - m[f"{c}_n"].astype(float)).abs()
        print(f"  max|d {c:16s}| = {d.max():.4f}   rows differing: {(d > 1e-4).sum()}")
    app_o = m.groupby("ID")["APP_o"].max()
    app_n = m.groupby("ID")["APP_n"].max()
    print(f"IDs whose final APP differs: {(app_o != app_n).sum()} / {len(app_o)} "
          f"(mean APP old {app_o.mean():.2f}, new {app_n.mean():.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
