import concurrent.futures
import datetime
import os

import pandas as pd
import rasterio
import rioxarray  # noqa: F401  # adds .rio accessor to xarray objects
import xarray as xr
from dateutil.relativedelta import relativedelta


def process_month(month_str, month_idx, data_file, grid_file, output_dir):
    """
    Process and resample data for a given month.

    Opens the precipitation and grid datasets, subsets to the given month,
    resamples using bilinear interpolation, and writes output to a NetCDF file.
    """
    # Open the precipitation data and grid dataset
    rpisco_ds = xr.open_dataset(data_file)
    rpisco_ds = rpisco_ds.rename({"pc": "Prec"})
    grid_ds = xr.open_dataset(grid_file)

    # Select the precipitation variable.
    # Note: Your dataset uses "Prec" (capitalized) as the variable name.
    if "Prec" in rpisco_ds.data_vars:
        rpisco_pp = rpisco_ds["Prec"]
    elif "pc" in rpisco_ds.data_vars:
        rpisco_pp = rpisco_ds["pc"]
    else:
        rpisco_pp = list(rpisco_ds.data_vars.values())[0]

    if rpisco_pp.rio.crs is None:
        rpisco_pp = rpisco_pp.rio.write_crs("EPSG:4326")

    # Ensure a time coordinate exists.
    # If "time" is not already a coordinate, check whether the file contains a "T" coordinate.
    if "time" not in rpisco_pp.coords:
        if "T" in rpisco_pp.coords:
            # Attach a new coordinate "time" along the "T" dimension using the existing T values.
            rpisco_pp = rpisco_pp.assign_coords(
                time=("T", rpisco_pp.coords["T"].values)
            )
        elif "z" in rpisco_pp.coords:
            num_layers = rpisco_pp.shape[0]
            seq_dates = pd.date_range(start="1981-01-01", periods=num_layers, freq="D")
            rpisco_pp = rpisco_pp.assign_coords(z=seq_dates).rename({"z": "time"})
        else:
            # Otherwise generate a new daily time coordinate if none exist.
            num_layers = rpisco_pp.shape[0]
            seq_dates = pd.date_range(start="1981-01-01", periods=num_layers, freq="D")
            rpisco_pp = rpisco_pp.assign_coords(time=("T", seq_dates))

    # Set the CRS if it is not already set.
    # In this example we assume EPSG:4326; change if needed.
    if rpisco_pp.rio.crs is None:
        rpisco_pp = rpisco_pp.rio.write_crs("EPSG:4326")

    # For the grid dataset, if it's a Dataset, select the first variable.
    if isinstance(grid_ds, xr.Dataset):
        grid_da = list(grid_ds.data_vars.values())[0]
    else:
        grid_da = grid_ds

    # Ensure the grid dataset also has a CRS.
    if grid_da.rio.crs is None:
        grid_da = grid_da.rio.write_crs("EPSG:4326")

    # Determine start and end dates for the month.
    date1 = pd.Timestamp(f"{month_str}-01")
    date2 = date1 + relativedelta(months=1)

    # Subset to the month using the condition time >= date1 and time < date2.
    ds_month = rpisco_pp.where(
        (rpisco_pp.time >= date1) & (rpisco_pp.time < date2), drop=True
    )

    # Only process if the monthly subset contains data.
    if ds_month.size > 0:
        # Resample the data to match the grid using bilinear interpolation.
        if ds_month.rio.crs is None:
            ds_month = ds_month.rio.write_crs("EPSG:4326")

        ds_month = ds_month.rename({"lat": "latitude"})
        ds_month = ds_month.rename({"lon": "longitude"})
        ds_month_resampled = ds_month.rio.reproject_match(
            grid_da, resampling=rasterio.enums.Resampling.bilinear
        )
        # Update attributes (similar to specifying units in the R code).
        ds_month_resampled.attrs["unit"] = "mm/day"

        # Define output path and write the resampled data as a NetCDF file.
        output_path = os.path.join(output_dir, f"prec_daily_{month_idx}.nc")
        ds_month_resampled.to_netcdf(output_path, mode="w")
        print(f"Processed {month_str} and saved to {output_path}")
    else:
        print(f"No data found for {month_str}")


def run_resampling(
    *,
    data_file: str | os.PathLike[str],
    grid_file: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    max_workers: int | None = None,
):
    """
    Runs the notebook's resampling pipeline (ProcessPoolExecutor over months).

    Notes
    -----
    This step is I/O heavy and each worker opens the NetCDF inputs. On machines with
    limited RAM, start with a small `max_workers` (e.g. 2–6) and scale up.
    """
    data_file = os.fspath(data_file)
    grid_file = os.fspath(grid_file)
    output_dir = os.fspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Open the precipitation dataset once to generate the overall time coordinate.
    rpisco_ds = xr.open_dataset(data_file)
    if "Prec" in rpisco_ds.data_vars:
        rpisco_pp = rpisco_ds["Prec"]
    elif "pc" in rpisco_ds.data_vars:
        rpisco_pp = rpisco_ds["pc"]
    else:
        rpisco_pp = list(rpisco_ds.data_vars.values())[0]

    # Using assign_coords in a robust way.
    if "time" not in rpisco_pp.coords:
        if "T" in rpisco_pp.coords:
            rpisco_pp = rpisco_pp.assign_coords(
                time=("T", rpisco_pp.coords["T"].values)
            )
            seq_dates = pd.to_datetime(rpisco_pp.coords["time"].values)
        elif "z" in rpisco_pp.coords:
            num_layers = rpisco_pp.shape[0]
            seq_dates = pd.date_range(start="1981-01-01", periods=num_layers, freq="D")
            rpisco_pp = rpisco_pp.assign_coords(z=seq_dates).rename({"z": "time"})
        else:
            num_layers = rpisco_pp.shape[0]
            seq_dates = pd.date_range(start="1981-01-01", periods=num_layers, freq="D")
            rpisco_pp = rpisco_pp.assign_coords(time=("T", seq_dates))
    else:
        seq_dates = pd.to_datetime(rpisco_pp.coords["time"].values)

    # Set the CRS for the overall dataset if needed.
    if rpisco_pp.rio.crs is None:
        rpisco_pp = rpisco_pp.rio.write_crs("EPSG:4326")

    # Create unique month labels for subsetting and filenames.
    unique_months = sorted({pd.Timestamp(date).strftime("%Y-%m") for date in seq_dates})
    unique_months_idx = [month.replace("-", "_") for month in unique_months]

    start_time = datetime.datetime.now()

    # Use ProcessPoolExecutor to process each month in parallel.
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for month_str, month_idx in zip(unique_months, unique_months_idx):
            future = executor.submit(
                process_month, month_str, month_idx, data_file, grid_file, output_dir
            )
            futures.append(future)

        # Handle exceptions from tasks.
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # Raises any exception encountered during processing.
            except Exception as exc:
                print(f"An exception occurred during processing: {exc}")

    end_time = datetime.datetime.now()
    print("Processing completed.")
    print("Total time elapsed:", end_time - start_time)
