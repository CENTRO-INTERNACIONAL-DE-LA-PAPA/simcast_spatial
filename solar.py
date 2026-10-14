"""
Vectorised sunrise / sunset (NOAA solar calculator algorithm).

This replaces the per-day ``astral`` calls used by the original SIMCAST
notebook.  For a single grid point and a season (~180 days) it is ~15x faster
than ``astral`` because the whole date vector is evaluated with NumPy at once.

Differences with the ``astral`` path
------------------------------------
* ``astral`` returns *civil* local time through a tz database, so it inherits
  historical daylight-saving periods (``America/Bogota`` had DST from
  1992-05-03 to 1993-04-04; ``America/Lima`` in 1986-87, 1990 and 1994).  The
  SIMCAST diurnal curve needs *solar* local time, so here the UTC offset is a
  fixed number of hours and no DST shift is ever applied.
* Outside DST periods both methods agree to well within one minute
  (see ``tests/test_solar.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_DEG = np.pi / 180.0

# Official zenith for sunrise/sunset (90 deg + 50 arc-min refraction/solar radius).
SUNRISE_ZENITH_DEG = 90.833


def _julian_century(dates: pd.DatetimeIndex) -> np.ndarray:
    # NOAA uses the Julian Day at *local noon*; the dependence on time of day
    # is negligible for sunrise/sunset (< 1 s) but we follow the reference.
    jd = dates.normalize().to_julian_date().to_numpy() + 0.5
    return (jd - 2451545.0) / 36525.0


def solar_declination_eot(dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Return (declination [deg], equation of time [minutes]) per date."""
    jc = _julian_century(dates)

    gmls = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360.0
    gmas = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    ecc = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)

    seoc = (
        np.sin(gmas * _DEG) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
        + np.sin(2 * gmas * _DEG) * (0.019993 - 0.000101 * jc)
        + np.sin(3 * gmas * _DEG) * 0.000289
    )
    true_long = gmls + seoc
    omega = 125.04 - 1934.136 * jc
    app_long = true_long - 0.00569 - 0.00478 * np.sin(omega * _DEG)

    mean_obliq = 23.0 + (
        26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0
    ) / 60.0
    obliq = mean_obliq + 0.00256 * np.cos(omega * _DEG)

    decl = np.degrees(np.arcsin(np.sin(obliq * _DEG) * np.sin(app_long * _DEG)))

    var_y = np.tan(obliq / 2.0 * _DEG) ** 2
    eot = 4.0 * np.degrees(
        var_y * np.sin(2 * gmls * _DEG)
        - 2 * ecc * np.sin(gmas * _DEG)
        + 4 * ecc * var_y * np.sin(gmas * _DEG) * np.cos(2 * gmls * _DEG)
        - 0.5 * var_y * var_y * np.sin(4 * gmls * _DEG)
        - 1.25 * ecc * ecc * np.sin(2 * gmas * _DEG)
    )
    return decl, eot


def sunrise_sunset_hours(
    lat: float,
    lon: float,
    dates,
    *,
    utc_offset_hours: float = -5.0,
    zenith_deg: float = SUNRISE_ZENITH_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sunrise and sunset as decimal local hours (e.g. 5.42 == 05:25).

    Parameters
    ----------
    lat, lon          : degrees (lon positive east, so Peru is negative)
    dates             : anything ``pd.DatetimeIndex`` accepts
    utc_offset_hours  : fixed offset of the local clock (Peru = -5)

    Returns
    -------
    (sunrise, sunset) float64 arrays, same length as ``dates``.
    Polar day/night (never in Peru) yields NaN.
    """
    dates = pd.DatetimeIndex(dates)
    decl, eot = solar_declination_eot(dates)

    lat_r = lat * _DEG
    decl_r = decl * _DEG
    cos_ha = np.cos(zenith_deg * _DEG) / (np.cos(lat_r) * np.cos(decl_r)) - np.tan(
        lat_r
    ) * np.tan(decl_r)
    with np.errstate(invalid="ignore"):
        ha_deg = np.degrees(np.arccos(cos_ha))  # NaN when |cos_ha| > 1

    noon_min = 720.0 - 4.0 * lon - eot + utc_offset_hours * 60.0
    sunrise = (noon_min - 4.0 * ha_deg) / 60.0
    sunset = (noon_min + 4.0 * ha_deg) / 60.0
    return sunrise, sunset


def standard_utc_offset_hours(timezone: str) -> float:
    """
    Standard (non-DST) UTC offset of an IANA zone, in hours.

    Evaluated on 2000-01-15, a date with no DST in either ``America/Lima`` or
    ``America/Bogota``.
    """
    import datetime as _dt
    import zoneinfo

    tz = zoneinfo.ZoneInfo(timezone)
    ref = _dt.datetime(2000, 1, 15, 12, 0, tzinfo=tz)
    off = ref.utcoffset()
    if off is None:
        raise ValueError(f"Could not determine UTC offset for {timezone!r}")
    return off.total_seconds() / 3600.0
