## simcast-spatial

This repo contains a **Python-only pipeline** (extracted from a Colab notebook) to:

1. Resample gridded climate rasters to a target grid (`resampling.py`)
2. Extract (“mask”) raster values to a point grid and write monthly Parquet files (`masking.py`)
3. Run the SIMCAST model per point/season using Dask (`load_variables.py` + `simcast.py`)
4. Build BU/FU rasters and an “all lands” animation (`animation_all_lands.py`)

The code is currently tuned to the original Peru potato-zoning workflow, but the goal is to make it reusable by **passing paths + resource settings from `main.py`** instead of editing module code.

## 2026 rerun (dew point extended to 2020, PISCOp v3.0)

The SIMCAST inputs are now built by **`prepare_inputs.py`** from these sources
(see `config.SOURCES_V3`, the default `EXAMPLE_SOURCES`), instead of the legacy
resampling + masking steps:

| variable | years | source | method |
|---|---|---|---|
| `tmax`, `tmin` | 1981–2020 | PISCOt v1.2 yearly NetCDFs (`_raw/v1_2/{tmax,tmin}/`) | exact grid-cell indexing |
| `td` | 1981–2016 | PISCOeo_pm monthly NetCDFs (`_raw/v1_1/td/`) | exact grid-cell indexing |
| `td` | 2017–2020 | TimesFM 3.0 predictions (`/home/cvelasquez/tdew_timesfm/preds_real/id_bucket=NNNN/pred.parquet`, column `pred`, `day` 0 = 2017‑01‑01, bucket = `gid % 8192`) | ID remap |
| `prec` | 1981–2025 | **PISCOp v3.0** daily 0.1° (`PISCOp_d.nc`, Gutierrez et al. 2026; time axis `Z1` = day index from 1981‑01‑01) | bilinear interpolation at the points |

Two input trees exist (`config.py`):

- `simcast_inputs_v2`: as above but `prec` from PISCOp v2 (1981–2019, old parquets re-written).
- `simcast_inputs_v3` (default): `prec` from PISCOp v3.0; `tmax`, `tmin`, `td` are
  symlinks to the v2 tree (`InputSources.link_from` / `link_variables`).

**Precipitation "resampling".** The legacy workflow resampled the 0.1° product to
the 0.01° grid with GDAL bilinear (`resampling.py`, ~350 MB NetCDF per month) and
then picked the node under each point (`masking.py`). `extract_prec_bilinear` does
both in one pass, interpolating straight at the 302 k points (2.4 GB cube read
once, ~4 min for 1981–2025). On PISCOp v2 it reproduces the old prec parquets to
< 2e‑5 mm. The only difference is the coastal mask: GDAL returned NaN when any of
the 4 neighbouring 0.1° cells was sea, and SIMCAST then filled those NaN with
**0 mm** (1 632 points, 0.5 %). The default `prec_nan_policy="renormalize"` drops
sea cells and rescales the remaining weights, so those points get real rainfall;
`"propagate"` reproduces the old mask.

PISCOp v3.0 vs v2 at the potato points (sample months 1985–2019): v3 is 7–30 %
drier on average, daily correlation 0.59–0.71, monthly-total correlation 0.86–0.95.

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

Coverage: with PISCOp v3.0 every season up to 2020 is complete. `cultivo1`
2020 (Nov 2020–Apr 2021) is not run because `tmax`/`tmin`/`td` end in 2020‑12.
With the v2 inputs, `prec` ended in 2019‑12, so `cultivo2`/`cultivo3` 2020 were
empty and `cultivo1` 2019 truncated. Each season folder contains a
`coverage.json` saying whether it was truncated.

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

`main.py` is the entry point. Paths live in `config.py`; steps and varieties
are chosen on the command line:

```bash
uv run python main.py --steps prepare
```

```bash
uv run python main.py --steps simcast rasters merge trend maps animations --vt r s
```

Steps: `prepare`, `resampling`, `masking` (legacy), `simcast`, `rasters`,
`merge`, `points`, `trend`, `maps`, `animations`. `--vt` takes the resistance classes
(`s`, `ms`, `mr`, `r`, `hr`); each one writes to `config.RESULTS_ROOTS[vt]`
(`results_resistant_v3`, `results_susceptible_v3`).

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
- every result table (`simcast_results.parquet`, `yearly_aggregated_data.parquet`,
  `statistical_trends_all.parquet`) carries `lon`, `lat` and `altitude` (m a.s.l.,
  column `Alt` of the potato grid) right after `ID` (`point_attrs.py`). Older
  result trees can be updated in place with `main.py --steps points --vt r s`.
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

- `compute_trend_stats(...)` writes a stats parquet (e.g. `statistical_trends_all.parquet`) with columns like `slope_app`, `p_value_app`, `std_err_app`, `conf_int_lower_app`, `conf_int_upper_app`, `r_squared_app`, `n_years_app` (and the same for `abu`/`afu`), one row per potato-grid ID, plus the yearly table (`yearly_aggregated_data.parquet`: `ID, lon, lat, altitude, year, period, region, APP, ABU, AFU, n_days`; `altitude` is the `Alt` column of the potato grid, m a.s.l.).
- `create_significance_map(...)` writes a PNG showing significant increase/decrease/not-significant for a chosen `variable_prefix` (`app`, `abu`, `afu`).
- `create_coefficient_map(...)` writes a PNG showing slope magnitude for significant points.

Changes in the Oct 2026 rewrite (see the module docstring):

- Every season file is read **once** and reduced to one row per ID (a process pool over files; the old code re-scanned all files for each batch of 500 IDs). The whole step now takes about a minute.
- The "year" of a metric is the **season** (the `<year>` folder), not the calendar year of each daily row. `cultivo1` runs Nov–Apr, so the legacy calendar-year aggregation split each season's BU/FU between two years and mixed the end of one season with the start of the next. `aggregate_by="calendar_year"` restores the legacy behaviour.
- Seasons whose `coverage.json` says `truncated` (e.g. `cultivo1` 2019, cut in Dec 2019 because `prec` stops there) are excluded by default (`exclude_truncated`).
- The OLS is computed in closed form with NumPy/SciPy for all IDs at once; it matches `statsmodels.OLS` to floating-point precision (`tests/test_trend.py`). `year_range` is inclusive at both ends (the legacy code excluded both endpoints).
- A metric that never changes over the years (e.g. `ABU == 0` every season at ~100 k cold Sierra Alta points with the resistant variety) is reported as slope 0 / p-value 1 ("no significant trend") instead of NaN ("no data").
- `statsmodels` is no longer required (only `scipy`); it is in the `dev` group for the equivalence test.

### Step 4 — BU/FU rasters

Uses `animation_all_lands.generate_bu_fu_rasters(...)` to write:

- `BU_sum_<year>_<period>_<region>.tif`
- `FU_sum_<year>_<period>_<region>.tif`

Rasters are built with the vectorised grid writer shared with `load_variables.create_raster` (pixel-identical to the previous `iterrows` version, ~2 s instead of ~40 s per raster), in parallel over season files, skipping rasters that already exist.

### Step 4b — Merged yearly rasters for ArcGIS

`uv run python main.py --steps merge --vt r s` (`merge_rasters.py`) writes, in
every `<results_root>/<year>/` folder, one GeoTIFF per variable joining the three
regional season rasters:

- `APP_<variety>_<year>.tif` (int16, max applications per season)
- `BU_<variety>_<year>.tif` (float32, season sum of blight units)
- `FU_<variety>_<year>.tif` (float32, season sum of fungicide units)

Same 0.01° grid as the inputs, EPSG:4326, NoData = -9999, no spaces in the
names. The layout is a plain GeoTIFF (striped, uncompressed, one page), like the
regional rasters: an earlier tiled + LZW + floating-point-predictor + internal
overviews version displayed as stripes with an extra small page in some viewers.
`merge_year(..., layout="cog")` still writes the compact variant. The regions never overlap (checked), so each
pixel keeps the value of its own region's season. The band description and the
metadata tags record the variable, units, variety and the regions merged.
Note that `cultivo1` (Sierra Alta) of `<year>` runs Nov `<year>`–Apr `<year+1>`,
and 2020 has no Sierra Alta because the climate inputs end in Dec 2020.

### Step 5 — “All lands” animations (APP, BU, FU)

`animation_all_lands.make_all_lands_animation(results_root=..., variable=...)`
writes one GIF and one MP4 per variable into the results root:

| variable | rasters used | output |
|---|---|---|
| `APP` | `simcast_<year>_<period>_<region>.tif` (max APP, written by the SIMCAST step) | `lateblight_all_lands_app.{gif,mp4}` |
| `BU` | `BU_sum_*.tif` | `lateblight_all_lands_bu.{gif,mp4}` |
| `FU` | `FU_sum_*.tif` | `lateblight_all_lands_fu.{gif,mp4}` |

Each frame is one year, adding the three cultivo rasters (every point belongs
to one region). Colours are discrete classes shared by all frames (see
"Colour scales" below); `classed=False` restores the old continuous scale. The lower-level helpers
(`collect_paths_by_year`, `build_all_lands_data`, `create_all_lands_animation`,
`save_all_lands_animation`) are unchanged.

Optional: boundary overlay via GADM (`animation_all_lands.load_gadm_peru`). A URL is downloaded once to `~/.cache/simcast_spatial/` and reused by every map; if you can’t access the URL, download the `.gpkg` and point to the local path. Years without any BU raster (e.g. 2020, which has no `prec`) are skipped.

### Step 6 — Response to El Niño Costero, season by season

`uv run python main.py --steps elnino --vt r s` (`el_nino_icen.py`) writes to
`<results_root>/el_nino_icen/`:

| file | content |
|---|---|
| `icen_slope_maps_<variety>.png` | APP, BU, FU: slope per point against the season-mean ICEN (SD of the variable per +1 ICEN), stippled where local p < 0.05, with the field significance in each panel |
| `icen_scatter_regions_<variety>.png` | per unit (SIMCAST region × Norte/Centro/Sur) and variable: season ICEN vs. unit anomaly, one dot per season coloured by ICEN category, fitted line, slope, r, p |
| `icen_event_maps_<variety>.png` | mean of the neutral seasons (ICEN −0.7 to 0.5), then one anomaly map per El Niño event (1982–83, 1986–87, 1991–92, 1997–98, 2015–16); each region shows the event's season with the highest ICEN, blank if that season was not warm |
| `icen_regional_<variety>.csv` | the data behind the scatter plots |
| `icen_summary_<variety>.json` | regional slopes, r, p and the field-significance results |

Method:

- **El Niño per season.** Each point has one season per year, fixed by its region: Sierra Media Apr–Jul, Costa y Valle Interandino Jul–Dec, Sierra Alta Nov–Apr. El Niño is measured as the mean of the official monthly ICEN (ENFEN 2024, `reference/icen_enfen2024.txt`) over that season's months. So Sierra Alta Nov 1998–Apr 1999 (ICEN −0.37) counts as neutral although 1998 belongs to the 1997–98 event.
- **Regression.** The season value is regressed on the season ICEN over all 39 seasons 1981–2019 (El Niño, La Niña and neutral), per point and per unit.
- **Significance.** The ICEN series is permuted among years in blocks of two consecutive years (5000 permutations), so a two-year event moves as one piece. The same permutation is applied to every point, which gives the field significance (Livezey & Chen 1983).
- **Exclusions.** 2020 is excluded because it has no Sierra Alta season.

## Colour scales (all maps, animations and figures)

The eye reads a continuous colour ramp as a handful of steps, not as a number,
so every map uses **discrete classes** with round, labelled breaks
(`viz_palette.py`):

| What is shown | Scale |
|---|---|
| A magnitude (APP, BU, FU, non-Niño mean) | one hue, light → dark blue, ~6 classes; first and last classes open-ended; a class of its own for exact zeros when they are ≥ 10 % of the map (BU) |
| A change (El Niño anomaly, trend per decade) | blue (less) ↔ grey (about zero) ↔ red (more), 4 classes per arm |
| Trend significance | red = significant increase, blue = significant decrease, light grey = no significant trend |

Breaks are equal round steps between the 1st and 99th percentiles, or rounded
quantiles when the variable is very skewed (median < 25 % of the 99th
percentile, e.g. BU). One set of classes is used for all frames of an animation
and all panels of a figure, so a colour means the same value everywhere. The red
ramp is lightness-matched to the blue ramp in OKLCH, so both arms of a diverging
scale weigh the same.

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
