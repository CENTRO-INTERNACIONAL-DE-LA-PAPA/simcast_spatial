import numpy as np
import pandas as pd
import pytest

from simcast import calculate_hhr, simcast_model


def _synthetic_climate(n=120, seed=0, start="2010-11-01"):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="D")
    tmin = rng.uniform(2, 9, n)
    return pd.DataFrame(
        {
            "FECHA": dates,
            "TMIN": tmin,
            "TMAX": tmin + rng.uniform(6, 14, n),
            "TDEW": tmin + rng.uniform(-3, 2, n),
            "PP": rng.exponential(2.5, n) * (rng.uniform(size=n) < 0.5),
        }
    )


def test_hhr_shape_and_columns():
    df = _synthetic_climate()
    out = calculate_hhr(df, lon=-74.5, lat=-12.3, sun_method="noaa")
    assert list(out.columns) == [
        "date", "hr_ge_thresh", "tmean_ge_thresh", "rain_mm", "tavg_C",
    ]
    assert len(out) == len(df) - 4  # first two + last two days dropped
    assert out["date"].iloc[0] == df["FECHA"].iloc[2]
    assert out["hr_ge_thresh"].between(0, 24).all()


def test_noaa_matches_astral_outside_dst():
    df = _synthetic_climate(n=150, start="2010-04-01")
    a = calculate_hhr(df, lon=-74.5, lat=-12.3, timezone="America/Lima", sun_method="astral")
    b = calculate_hhr(df, lon=-74.5, lat=-12.3, timezone="America/Lima", sun_method="noaa")
    # sunrise/sunset differ by < 1 minute -> hourly temps differ by ~0.01 C
    assert np.abs(a["tmean_ge_thresh"] - b["tmean_ge_thresh"]).max() < 0.05
    assert np.abs(a["hr_ge_thresh"].astype(int) - b["hr_ge_thresh"].astype(int)).max() <= 1
    assert (a["hr_ge_thresh"] != b["hr_ge_thresh"]).mean() < 0.05


def test_unknown_sun_method():
    with pytest.raises(ValueError):
        calculate_hhr(_synthetic_climate(), lon=-74.5, lat=-12.3, sun_method="x")


def test_short_input_returns_empty():
    out = calculate_hhr(_synthetic_climate(n=3), lon=-74.5, lat=-12.3)
    assert out.empty


def test_simcast_model_applications():
    df = _synthetic_climate(n=150)
    hhr = calculate_hhr(df, lon=-74.5, lat=-12.3)
    res = simcast_model(hhr, "r", min_day=5, forced_day=25)
    assert {"BU", "FU", "ABU", "AFU", "APP"} <= set(res.columns)
    assert res["APP"].is_monotonic_increasing
    # forced application on day index 25
    assert res["APP"].iloc[25] >= 1
    assert res["APP"].iloc[24] == 0 or res["APP"].iloc[24] <= res["APP"].iloc[25]
