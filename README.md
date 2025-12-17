## simcast-spatial

This repo contains a **Python-only pipeline** (extracted from a Colab notebook) to:

1. Resample gridded climate rasters to a target grid (`resampling.py`)
2. Extract (“mask”) raster values to a point grid and write monthly Parquet files (`masking.py`)
3. Run the SIMCAST model per point/season using Dask (`load_variables.py` + `simcast.py`)
4. Build BU/FU rasters and an “all lands” animation (`animation_all_lands.py`)

The code is currently tuned to the original Peru potato-zoning workflow, but the goal is to make it reusable by **passing paths + resource settings from `main.py`** instead of editing module code.

## Setup

Install dependencies from `pyproject.toml`:

- With `uv`: `uv sync` then `uv run python main.py`
- Without `uv`: create a venv and install the packages listed under `dependencies` in `pyproject.toml`

Run from the repo root so imports like `import resampling` work.

## Data you need

### Spatial grid (points)

- A GeoPackage/Shapefile readable by GeoPandas, with geometries (points or polygons) and an `ID` column (or it will be created from index).
- The notebook’s default region split expects a `Type` column with Peru-specific category strings (see `load_variables.py#split_regions_by_type`).

### NetCDF inputs

- **Precipitation “full” dataset** (used only for resampling): `data.nc` (or equivalent) containing `pc`/`Prec` and a time-like dimension/coord.
- **Target grid NetCDF**: a file whose grid you want to match (e.g. a monthly `tmin` file).
- **Monthly NetCDFs per variable** for masking/extraction:
  - `prec_daily_YYYY_MM.nc` (output of resampling)
  - `tmax_daily_YYYY_MM.nc`
  - `tmin_daily_YYYY_MM.nc`
  - `td_daily_YYYY_MM.nc`

### Parquet layout (required by SIMCAST step)

`load_variables.py` expects monthly Parquets in this layout:

`<PARQUET_BASE_DIR>/<variable>/Outputs/<variable>_daily_YYYY_MM.parquet`

Each Parquet must contain at least: `ID`, `FECHA`, `Value`.

## Running the pipeline

`main.py` is the example entry point. Edit the paths + toggles there and run:

- `python main.py`

### Step 1 — Resampling (prec only)

Uses `resampling.run_resampling(...)` to write:

- `prec_daily_YYYY_MM.nc` into your chosen output directory.

### Step 2 — Masking / extraction (NetCDF → Parquet)

Uses `masking.extract_folder_to_parquets(...)` to write monthly Parquet files per variable.

Tune:

- `chunk_size` (smaller = less RAM, slower)
- `max_workers` (more = faster, more RAM/I/O)

### Step 3 — SIMCAST (Parquet → results)

Uses `load_variables.parallel_simcast_pipeline(...)` to write:

- `simcast_results.parquet` per `(year, cultivo, region)`
- optionally `simcast_<year>_<period>_<region>.tif` (APP raster)

Tune:

- `n_workers`, `threads_per_worker`
- `memory_limit`
- `batch_size` (IDs per in-memory batch)

#### How Dask speeds this up

`parallel_simcast_pipeline(...)` uses **Dask Distributed** to parallelize across independent seasonal runs:

- Builds one coarse task per `(year, region, period)` using `@dask.delayed process_region_period(...)`.
- Submits tasks with `client.compute(...)` and streams completed results via `as_completed(...)`, so it can write outputs incrementally instead of holding everything in RAM.
- Scatters the (static) point grid once (`client.scatter(potato_grid, broadcast=True)`) so each worker can reuse it without re-serializing for every task. This reduces scheduler/serialization overhead and keeps the task graph smaller (important when the grid is large and reused by many tasks).

Inside each task, the heavy work is still local CPU compute: reading monthly Parquets for the date window (with PyArrow filter pushdown), merging variables, then running `calculate_hhr(...)` and `simcast_model(...)` per `ID` in batches (`batch_size`) to control peak memory.

#### Multi-computer Dask (suggested; not implemented in code)

For multiple machines, you typically run a Dask **scheduler** on one host and **workers** on the others, then connect from Python.

1) Start the scheduler (on one machine):

```bash
dask scheduler --host 0.0.0.0 --port 8786 --dashboard-address :8787
```

2) Start one worker process per machine (or more, depending on RAM), pointing at the scheduler:

```bash
dask worker tcp://SCHEDULER_IP:8786 --nthreads 4 --memory-limit 14GB
```

3) Connect from Python:

```python
from dask.distributed import Client

client = Client("tcp://SCHEDULER_IP:8786")
print(client)
```

Notes for multi-machine runs:

- All worker machines must see the same `parquet_base_dir` and `results_root` paths (usually via a shared filesystem like NFS).
- Open firewall ports for the scheduler (`8786`) and dashboard (`8787`) if you want the web UI.
- Ensure the same Python environment (same package versions) is available on every worker machine.

This repo’s `parallel_simcast_pipeline(...)` currently creates its own local `Client(...)`. A common adaptation is to add an optional `scheduler_address` or `client` parameter and skip local cluster creation when provided (left as a suggested change; not implemented here).

### Step 3b — Trend significance (OLS regression over years)

This step runs a per-grid-point linear regression over time using the SIMCAST outputs.

It reads all `simcast_results.parquet` under `results_root`, aggregates to **one row per (ID, year)**, then fits:

- `APP ~ year` where `APP` is `max(APP)` per year
- `ABU ~ year` where `ABU` is `sum(BU)` per year
- `AFU ~ year` where `AFU` is `sum(FU)` per year

Code lives in `trend_significance.py`:

- `compute_trend_stats(...)` writes a stats parquet (e.g. `statistical_trends_all.parquet`) with columns like `slope_app`, `p_value_app`, `r_squared_app` (and the same for `abu`/`afu`).
- `create_significance_map(...)` writes a PNG showing significant increase/decrease/not-significant for a chosen `variable_prefix` (`app`, `abu`, `afu`).
- `create_coefficient_map(...)` writes a PNG showing slope magnitude for significant points.

Dependencies:

- `compute_trend_stats(...)` requires `statsmodels` (for OLS). The plotting functions do not.

Dask notes:

- This also uses Dask Distributed (tasks are batches of IDs). It scatters the file list once to reduce serialization overhead, then streams results back and writes the final parquet(s).
- Tune `batch_size`, `n_workers`, `threads_per_worker`, and `memory_limit` the same way as the SIMCAST step.

### Step 4 — BU/FU rasters

Uses `animation_all_lands.generate_bu_fu_rasters(...)` to write:

- `BU_sum_<year>_<period>_<region>.tif`
- `FU_sum_<year>_<period>_<region>.tif`

### Step 5 — “All lands” animation

Uses:

- `collect_paths_by_year(...)`
- `build_all_lands_data(...)`
- `create_all_lands_animation(...)`
- `save_all_lands_animation(...)`

Optional: boundary overlay via GADM (`animation_all_lands.load_gadm_peru`). If you can’t access the URL, download the `.gpkg` and point to the local path.

## Resource notes (CPU/RAM)

- This pipeline is **CPU-bound**; `simcast.py` uses Numba to JIT-compile inner loops for CPU speedups (no CUDA/GPU code path).
- Resampling + masking can spawn many processes; on machines with limited RAM, start with a small worker count (2–6).
- The SIMCAST step can be memory-heavy when merging climate variables. Reduce `batch_size` and Dask worker counts if you see worker deaths/spilling.
