## simcast-spatial

This repo contains a **Python-only pipeline** (extracted from a Colab notebook) to:

1. Resample gridded climate rasters to a target grid (`resampling.py`)
2. Extract (“mask”) raster values to a point grid and write monthly Parquet files (`masking.py`)
3. Run the SIMCAST model per point/season using Dask (`load_variables.py` + `simcast.py`)
4. Build BU/FU rasters and an “all lands” animation (`animation_all_lands.py`)

The code is currently tuned to the original Peru potato-zoning workflow, but the goal is to make it reusable by **passing paths + resource settings from `main.py`** instead of editing module code.

## 2026 rerun (dew point extended to 2020)

The SIMCAST inputs are now built by **`prepare_inputs.py`** from these sources
(see `config.EXAMPLE_SOURCES`), instead of the legacy resampling + masking steps:

| variable | years | source | method |
|---|---|---|---|
| `tmax`, `tmin` | 1981–2020 | PISCOt v1.2 yearly NetCDFs (`_raw/v1_2/{tmax,tmin}/`) | exact grid-cell indexing |
| `td` | 1981–2016 | PISCOeo_pm monthly NetCDFs (`_raw/v1_1/td/`) | exact grid-cell indexing |
| `td` | 2017–2020 | TimesFM 3.0 predictions (`/home/cvelasquez/tdew_timesfm/preds_real/id_bucket=NNNN/pred.parquet`, column `pred`, `day` 0 = 2017‑01‑01, bucket = `gid % 8192`) | ID remap |
| `prec` | 1981–2019 | existing `prec/Outputs/*.parquet` (PISCOp resampled bilinearly) | re-written |

Every potato point sits exactly on a node of the 0.01° PISCO grid, so values are
read as `var[t, row, col]` (no interpolation). This reproduces bit-for-bit the
parquets previously produced with `masking.py` (checked for `tmin` 2020‑12 and
`td` 2016‑12). The mapping potato `ID` ↔ PISCO cell is saved in
`<parquet_base_dir>/grid_mapping.parquet` and all sources in `provenance.json`.

Output parquets are **sorted by `(ID, FECHA)`** with ~5 000 IDs per row group and
`zstd` compression (14 MB/month instead of 56 MB), so the ID-range filters in
`load_variables.load_variable_data` prune row groups through parquet statistics.

Run it with:

```bash
uv run python prepare_inputs.py --workers 8
```

(idempotent: existing monthly files are skipped; ~15 min for all variables).

Known gaps: `prec` ends in 2019‑12, so the 2020 seasons (`cultivo2`/`cultivo3`
2020) and the tail of `cultivo1` 2019 (Jan–Apr 2020) have no precipitation and
are skipped/truncated. Each season folder now contains a `coverage.json` saying
whether it was truncated.

> The 2017–2020 dew point is a **model prediction without observed truth**
> (TimesFM 3.0 weights are non-commercial; research deliverable, not
> operational). Results for those years should be labelled accordingly.

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

- `simcast_results.parquet` per `(year, cultivo, region)` (sorted by `ID, date`)
- `coverage.json` per season (requested vs. actual date range, number of IDs)
- optionally `simcast_<year>_<period>_<region>.tif` (APP raster)

Tune:

- `n_workers`, `threads_per_worker` (their product = concurrent chunks)
- `memory_limit` (per worker process)
- `id_chunk_size` (IDs per Dask task; ~0.5 GB RAM per concurrent 5k-ID chunk)
- `skip_existing=True` resumes a run (seasons with results are not recomputed)
- `sun_method="noaa"` (vectorised sunrise/sunset) or `"astral"` (original)
- `client=` / `scheduler_address=` to reuse an existing or multi-machine cluster

#### How Dask speeds this up

`parallel_simcast_pipeline(...)` uses **Dask Distributed**:

- Every `(year, region, period)` season is split into chunks of `id_chunk_size`
  points; each chunk is an independent `@dask.delayed process_id_chunk(...)` task.
  Only the small `[ID, lon, lat]` table of the chunk travels to the worker.
- Each task reads **only its own IDs** from the monthly Parquets (ID range +
  `in` filters, with row-group pruning), merges the four variables, then runs
  `calculate_hhr(...)` and `simcast_model(...)` per `ID`.
- Completed chunks stream back via `as_completed(...)`; when all chunks of a
  season are in, the season is concatenated and written, so peak RAM in the
  driver is one season at a time.

#### Where the time goes (and why there is no CUDA path)

Per point and season (~180 days) the cost is roughly:

| step | original | now |
|---|---:|---:|
| sunrise/sunset (`astral`, 2 calls/day) | 3.6 ms | 0.2 ms (NOAA, vectorised) |
| diurnal curve + RH + HHR (NumPy) | 1.8 ms | 1.8 ms |
| BU/FU tables + application scan (Numba) | 0.4 ms | 0.4 ms |

≈ 2.5 ms per point-season. Measured end to end (Parquet I/O + Dask) one full
year (302 k points, 3 seasons) takes **66 s with 20 single-threaded workers**
(`id_chunk_size=5000`, `sun_method="noaa"`), so the whole 1981–2020 run is
about 45 minutes on the 32-core workstation. The same year took 356 s with
`astral` and 10 workers × 2 threads: the per-point work is pandas/Python
bound, so threads inside one process serialise on the GIL. The model kernels are tiny decision tables
with a sequential daily scan per point, so a GPU would be idle waiting for data:
CUDA is not worth it here. If more speed is ever needed, the next step is to
evaluate `calculate_hhr` for a whole chunk at once (2-D `(points, days)` arrays)
instead of one point at a time.

#### Sunrise/sunset note

The original code called `astral` with `pytz.timezone("America/Bogota")`.
That returns *civil* time, which inherits the DST period Colombia observed from
1992‑05‑03 to 1993‑04‑04 (and Peru's own DST years if `America/Lima` is used):
during those months the diurnal curve was shifted by one hour. The default
`sun_method="noaa"` uses a fixed UTC‑5 offset (solar local time) and agrees
with `astral` to < 1 minute outside DST periods (`tests/test_solar.py`).

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

`parallel_simcast_pipeline(...)` accepts `scheduler_address="tcp://SCHEDULER_IP:8786"` (or an existing `client=`) and then skips creating a local cluster.

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

- This pipeline is **CPU/I-O-bound**; `simcast.py` uses Numba to JIT-compile inner loops and NumPy for the hourly reconstruction (no CUDA/GPU code path, see "Where the time goes" above).
- `prepare_inputs.py` reads one month of NetCDF (~350 MB) per worker; 8 workers need ~8 GB.
- The SIMCAST step holds one chunk of `id_chunk_size` points per worker (~0.5 GB for 5 000 points). Reduce `id_chunk_size` or `n_workers` if you see worker deaths/spilling; keep `threads_per_worker=1`.

## Tests

```bash
uv run --group dev pytest -q
```

`tests/test_solar.py` checks the vectorised sunrise/sunset against `astral`,
`tests/test_prepare_inputs.py` runs the ETL on a tiny synthetic grid, and
`tests/test_simcast.py` covers `calculate_hhr` / `simcast_model`.
`scripts/compare_results.py OLD_ROOT NEW_ROOT` compares two result trees season by season.
