import datetime as dt

import numpy as np
import pandas as pd
import pytest
from astral import LocationInfo
from astral.sun import sunrise, sunset

from solar import standard_utc_offset_hours, sunrise_sunset_hours


def _astral_hours(lat, lon, dates, tz="America/Lima"):
    loc = LocationInfo(latitude=lat, longitude=lon)
    sr, ss = [], []
    for d in dates:
        a = sunrise(loc.observer, date=d.date(), tzinfo=tz)
        b = sunset(loc.observer, date=d.date(), tzinfo=tz)
        sr.append(a.hour + a.minute / 60 + a.second / 3600)
        ss.append(b.hour + b.minute / 60 + b.second / 3600)
    return np.array(sr), np.array(ss)


@pytest.mark.parametrize(
    "lat,lon",
    [(-12.05, -77.04), (-4.5, -69.0), (-18.2, -70.3), (-9.0, -78.0), (-13.5, -72.0)],
)
def test_matches_astral_outside_dst(lat, lon):
    # Years without DST in America/Lima.
    dates = pd.date_range("2000-01-01", "2020-12-31", freq="11D")
    a_sr, a_ss = _astral_hours(lat, lon, dates)
    n_sr, n_ss = sunrise_sunset_hours(lat, lon, dates, utc_offset_hours=-5.0)
    assert np.max(np.abs(a_sr - n_sr)) * 60 < 1.0  # minutes
    assert np.max(np.abs(a_ss - n_ss)) * 60 < 1.0


def test_dst_period_is_ignored_by_noaa():
    # America/Bogota observed DST 1992-05-03 .. 1993-04-04. astral shifts by
    # +1h there; the NOAA path must not.
    lat, lon = -12.3, -74.5
    before = pd.DatetimeIndex([dt.datetime(1992, 4, 20)])
    inside = pd.DatetimeIndex([dt.datetime(1992, 5, 20)])
    a_before, _ = _astral_hours(lat, lon, before, tz="America/Bogota")
    a_inside, _ = _astral_hours(lat, lon, inside, tz="America/Bogota")
    n_before, _ = sunrise_sunset_hours(lat, lon, before)
    n_inside, _ = sunrise_sunset_hours(lat, lon, inside)
    assert abs((a_inside - a_before)[0]) > 0.9  # astral jumps ~1 h
    assert abs((n_inside - n_before)[0]) < 0.2  # NOAA drifts a few minutes


def test_standard_offsets():
    assert standard_utc_offset_hours("America/Lima") == -5.0
    assert standard_utc_offset_hours("America/Bogota") == -5.0
    assert standard_utc_offset_hours("UTC") == 0.0
