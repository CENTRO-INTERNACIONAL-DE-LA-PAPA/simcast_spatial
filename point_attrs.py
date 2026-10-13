"""
Point coordinates and attributes (lon, lat, altitude) for SIMCAST outputs.

``point_attributes`` builds the table from the potato grid file;
``add_point_columns`` adds those columns to any table with an ``ID`` column
(pandas or pyarrow); ``backfill_results`` rewrites existing
``simcast_results.parquet`` files in place with the extra columns.
"""

from __future__ import annotations

import concurrent.futures as cf
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

POINT_COLUMNS = ("lon", "lat", "altitude")
DEFAULT_ALTITUDE_COLUMN = "Alt"  # metres a.s.l. in the potato grid file


def point_attributes(
    potato_grid,
    *,
    altitude_column: str = DEFAULT_ALTITUDE_COLUMN,
) -> pd.DataFrame:
    """[ID, lon, lat, altitude] from a potato grid GeoDataFrame or file path."""
    if isinstance(potato_grid, (str, os.PathLike)):
        import geopandas as gpd

        potato_grid = gpd.read_file(os.fspath(potato_grid))
    ids = potato_grid["ID"].to_numpy() if "ID" in potato_grid.columns else potato_grid.index.to_numpy()
    geom = potato_grid.geometry.apply(lambda g: g if g.geom_type == "Point" else g.centroid)
    out = pd.DataFrame(
        {"ID": ids.astype(np.int64), "lon": geom.x.to_numpy(), "lat": geom.y.to_numpy()}
    )
    if altitude_column in potato_grid.columns:
        out["altitude"] = potato_grid[altitude_column].to_numpy()
    return out.sort_values("ID", kind="stable").reset_index(drop=True)


def _lookup(attrs: pd.DataFrame, ids: np.ndarray) -> dict[str, np.ndarray]:
    a = attrs.set_index("ID")
    pos = a.index.get_indexer(ids)
    if (pos < 0).any():
        missing = np.unique(ids[pos < 0])
        raise KeyError(f"{missing.size} IDs not in the point table (e.g. {missing[:3]})")
    return {c: a[c].to_numpy()[pos] for c in POINT_COLUMNS if c in a.columns}


def add_point_columns(table, attrs: pd.DataFrame):
    """
    Return ``table`` with ``lon``, ``lat``, ``altitude`` placed right after ``ID``.

    Works for a pandas DataFrame or a pyarrow Table; existing point columns are
    replaced.
    """
    if isinstance(table, pa.Table):
        ids = table.column("ID").to_numpy()
        cols = _lookup(attrs, ids)
        names = [n for n in table.column_names if n not in POINT_COLUMNS]
        rest = [n for n in names if n != "ID"]
        arrays = [table.column("ID")] + [pa.array(v) for v in cols.values()] + [table.column(n) for n in rest]
        return pa.Table.from_arrays(arrays, names=["ID", *cols.keys(), *rest])

    df = table.drop(columns=[c for c in POINT_COLUMNS if c in table.columns])
    cols = _lookup(attrs, df["ID"].to_numpy())
    rest = [c for c in df.columns if c != "ID"]
    out = df[["ID"]].copy()
    for k, v in cols.items():
        out[k] = v
    return pd.concat([out, df[rest]], axis=1)


def _backfill_one(path: str, attrs: pd.DataFrame, force: bool) -> str | None:
    pf = pq.ParquetFile(path)
    if not force and all(c in pf.schema_arrow.names for c in POINT_COLUMNS):
        return None
    n_rg = pf.metadata.num_row_groups
    rg_rows = pf.metadata.row_group(0).num_rows if n_rg else None
    meta = pf.schema_arrow.metadata
    table = pf.read()
    new = add_point_columns(table, attrs)
    if meta:
        # drop the stale pandas metadata (column list changed); keep others
        meta = {k: v for k, v in meta.items() if k != b"pandas"}
        new = new.replace_schema_metadata(meta or None)
    tmp = path + ".tmp"
    pq.write_table(new, tmp, row_group_size=rg_rows, compression="snappy")
    if pq.ParquetFile(tmp).metadata.num_rows != table.num_rows:
        os.remove(tmp)
        raise RuntimeError(f"row count changed while rewriting {path}")
    os.replace(tmp, path)
    return path


def backfill_results(
    results_root: str | os.PathLike[str],
    potato_grid_file: str | os.PathLike[str],
    *,
    parquet_name: str = "simcast_results.parquet",
    workers: int = 6,
    force: bool = False,
) -> list[str]:
    """Add lon/lat/altitude to every season parquet under ``results_root`` (in place)."""
    attrs = point_attributes(potato_grid_file)
    files = sorted(glob.glob(os.path.join(os.fspath(results_root), "*", "*", "*", parquet_name)))
    done = []
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_backfill_one, f, attrs, force): f for f in files}
        for fut in cf.as_completed(futs):
            r = fut.result()
            if r:
                done.append(r)
    print(f"[points] {Path(results_root)}: {len(done)} of {len(files)} files updated")
    return done
