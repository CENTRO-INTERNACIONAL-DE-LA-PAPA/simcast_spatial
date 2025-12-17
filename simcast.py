import numba as nb
import numpy as np
import pandas as pd
import pytz
from astral import LocationInfo
from astral.sun import sunrise, sunset
from numba import njit

# ------------------------------------------------------------------
# 1.  Vectorised BU / FU kernels
# ------------------------------------------------------------------

VT_INDEX = {"s": 0, "ms": 1, "mr": 2, "r": 3, "hr": 4}
BU_THRESH = np.array([30, 35, 40, 45, 50], dtype=np.int16)
FU_THRESH = np.array([15, 20, 25, 35, 45], dtype=np.int16)


@njit(cache=True, fastmath=True)
def calc_bu_vec(hhr, tavg, vt_idx, out):
    """
    Vectorised blight-unit decision table.

    Parameters
    ----------
    hhr   : 1-D float (hours RH>90 %)
    tavg  : 1-D float (average temp)
    vt_idx: int   (0..4 for s .. hr)
    out   : 1-D int  (pre-allocated result)
    """
    n = hhr.size
    for i in range(n):
        h = hhr[i]
        t = tavg[i]
        bu = 0

        # ---- 22.5 < T <= 27 ---------------------------------------
        if 22.5 < t <= 27.0:
            if vt_idx == 0:  # susceptible
                if h <= 6:
                    bu = 0
                elif h <= 9:
                    bu = 1
                elif h <= 12:
                    bu = 2
                elif h <= 15:
                    bu = 3
                elif h <= 18:
                    bu = 4
                else:
                    bu = 5
            elif vt_idx == 1:  # ms
                if h <= 9:
                    bu = 0
                elif h <= 18:
                    bu = 1
                else:
                    bu = 2
            else:  # mr/r/hr
                if h <= 15:
                    bu = 0
                else:
                    bu = 1

        # ---- 12.5 < T <= 22.5 -------------------------------------
        elif 12.5 < t <= 22.5:
            if vt_idx == 0:
                if h <= 6:
                    bu = 0
                elif h <= 9:
                    bu = 5
                elif h <= 12:
                    bu = 6
                else:
                    bu = 7
            elif vt_idx == 1:
                if h <= 6:
                    bu = 0
                elif h == 7:
                    bu = 1
                elif h == 8:
                    bu = 2
                elif h == 9:
                    bu = 3
                elif h == 10:
                    bu = 4
                elif h <= 12:
                    bu = 5
                else:
                    bu = 6
            else:
                if h <= 6:
                    bu = 0
                elif h == 7:
                    bu = 1
                elif h == 8:
                    bu = 2
                elif h == 9:
                    bu = 3
                elif h <= 12:
                    bu = 4
                else:
                    bu = 5

        # ---- 7.5 < T <= 12.5 --------------------------------------
        elif 7.5 < t <= 12.5:
            if vt_idx == 0:
                if h <= 6:
                    bu = 0
                elif h == 7:
                    bu = 1
                elif h <= 9:
                    bu = 2
                elif h == 10:
                    bu = 3
                elif h <= 12:
                    bu = 4
                elif h <= 15:
                    bu = 5
                elif h <= 24:
                    bu = 6
            elif vt_idx == 1:
                if h <= 6:
                    bu = 0
                elif h <= 9:
                    bu = 1
                elif h <= 12:
                    bu = 2
                elif h <= 15:
                    bu = 3
                elif h <= 18:
                    bu = 4
                else:
                    bu = 5
            else:
                if h <= 9:
                    bu = 0
                elif h <= 12:
                    bu = 1
                elif h <= 15:
                    bu = 2
                else:
                    bu = 3

        # ---- 3 <= T <= 7.5 ----------------------------------------
        elif 3.0 <= t <= 7.5:
            if vt_idx == 0:
                if h <= 9:
                    bu = 0
                elif h <= 12:
                    bu = 1
                elif h <= 15:
                    bu = 2
                elif h <= 18:
                    bu = 3
                else:
                    bu = 4
            elif vt_idx == 1:
                if h <= 12:
                    bu = 0
                else:
                    bu = 1
            else:
                if h <= 18:
                    bu = 0
                else:
                    bu = 1

        out[i] = bu


@njit(cache=True, fastmath=True)
def calc_fu_vec(rain, dsa, out):
    """
    Vectorised fungicide-unit table.
    rain : 1-D float (mm)
    dsa  : 1-D int   (days since application)
    """
    n = rain.size
    for i in range(n):
        r = rain[i]
        d = dsa[i]
        fu = 0

        if 0 < r < 1:
            fu = 1
        else:
            if d == 1:
                if r <= 1.45:
                    fu = 4
                elif r <= 3.45:
                    fu = 5
                elif r <= 6.00:
                    fu = 6
                else:
                    fu = 7
            elif d == 2:
                if r <= 1.45:
                    fu = 3
                elif r <= 4.45:
                    fu = 4
                elif r <= 8.00:
                    fu = 5
                else:
                    fu = 6
            elif d == 3:
                if r <= 2.45:
                    fu = 3
                elif r <= 5.00:
                    fu = 4
                else:
                    fu = 5
            elif 4 <= d <= 5:
                if r <= 2.45:
                    fu = 3
                elif r <= 8.00:
                    fu = 4
                else:
                    fu = 5
            elif 6 <= d <= 9:
                if r <= 4.00:
                    fu = 3
                else:
                    fu = 4
            elif 10 <= d <= 14:
                if r <= 1.45:
                    fu = 2
                elif r <= 8.00:
                    fu = 3
                else:
                    fu = 4
            elif d > 14:
                if r <= 8.00:
                    fu = 2
                else:
                    fu = 3
        out[i] = fu


# ───────────────────────────────────────────────────────────────
# Helper: broadcast column
# ───────────────────────────────────────────────────────────────
def _col(x: np.ndarray) -> np.ndarray:
    return x[:, None]  # (n, 1)


# ───────────────────────────────────────────────────────────────
# 1. García‑1 diurnal temperature curve  (bug‑free version)
# ───────────────────────────────────────────────────────────────
def diurnal_temp(
    Tn: np.ndarray, Tx: np.ndarray, Tp: np.ndarray, Hn: np.ndarray, Ho: np.ndarray
) -> np.ndarray:
    """
    Returns an (n, 24) array with hourly temperatures (hours 1..24).
    """
    n = Tn.size
    hrs_vec = np.arange(1, 25, dtype=np.float32)  # (24,)
    hrs_mat = np.tile(hrs_vec, (n, 1))  # (n, 24)

    Hx = Ho - 4
    Hp = Hn + 24

    # Transition temps
    To = Tx - 0.39 * (Tx - Tp)
    Tn_prev = np.roll(Tn, 1)
    Tx_prev = np.roll(Tx, 1)
    To_prev = Tx_prev - 0.39 * (Tx_prev - Tn_prev)
    Ho_prev = np.roll(Ho, 1)
    Hp_prev = np.roll(Hn, 1) + 24

    alpha = Tx - Tn
    r = Tx - To
    beta1 = (Tp - To) / np.sqrt(Hp - Ho)
    beta2 = (Tn - To_prev) / np.sqrt(Hp_prev - Ho_prev)

    # ── Full‑grid expressions (shape (n, 24)) ──────────────────────
    expr1 = _col(Tn) + _col(alpha) * ((hrs_mat - _col(Hn)) / (_col(Hx) - _col(Hn))) * (
        np.pi / 2.0
    )

    expr2 = _col(To) + _col(r) * np.sin(
        (np.pi / 2) + ((hrs_mat - _col(Hx)) / 4.0) * (np.pi / 2)
    )

    expr3 = _col(To) + _col(beta1) * np.sqrt(np.maximum(hrs_mat - _col(Ho), 0.0))

    expr4 = _col(To_prev) + _col(beta2) * np.sqrt(
        np.maximum(hrs_mat + 24 - _col(Ho_prev), 0.0)
    )

    # ── Assemble the 4 segments ───────────────────────────────────
    seg1 = (hrs_mat > _col(Hn)) & (hrs_mat <= _col(Hx))  # sunrise→Hx
    seg2 = (hrs_mat > _col(Hx)) & (hrs_mat <= _col(Ho))  # Hx→sunset
    seg3 = (hrs_mat > _col(Ho)) & (hrs_mat <= 24)  # sunset→24 h
    seg4 = (hrs_mat >= 1) & (hrs_mat <= _col(Hn))  # 0→sunrise

    T = np.empty((n, 24), dtype=np.float32)
    T[seg1] = expr1[seg1]
    T[seg2] = expr2[seg2]
    T[seg3] = expr3[seg3]
    T[seg4] = expr4[seg4]

    return T


def calculate_hhr(
    climdata: pd.DataFrame,
    lon: float,
    lat: float,
    rh_thresh: float = 90.0,
    timezone: str = "America/Bogota",
) -> pd.DataFrame:
    """
    Hour‑by‑hour humidity model (García‑1) with flexible RH threshold
    and 13:00–12:00 observation window.

    Parameters
    ----------
    climdata  : DataFrame with columns [FECHA, TMIN, TMAX, TDEW, PP]
    lon, lat  : point coordinates (float)
    rh_thresh : RH % threshold (default 90)

    Returns
    -------
    DataFrame  [date, hr_ge_thresh, tmean_ge_thresh, rain_mm, tavg_C]
               (first TWO and last TWO calendar days are dropped,
                matching the behaviour of the original R script).
    """
    if np.isnan(lon) or np.isnan(lat) or climdata.shape[0] < 5:
        return pd.DataFrame(
            columns=["date", "hr_ge_thresh", "tmean_ge_thresh", "rain_mm", "tavg_C"]
        )

    df = climdata.copy()
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    df["tavg_C"] = (df["TMIN"] + df["TMAX"]) / 2.0

    # --- sunrise / sunset in decimal hours ----------------------------
    loc = LocationInfo(latitude=lat, longitude=lon)
    tz = pytz.timezone(timezone)

    sr = df["FECHA"].apply(lambda d: sunrise(loc.observer, date=d, tzinfo=tz))
    ss = df["FECHA"].apply(lambda d: sunset(loc.observer, date=d, tzinfo=tz))

    Hn = (sr.dt.hour + sr.dt.minute / 60 + sr.dt.second / 3600).to_numpy(np.float32)
    Ho = (ss.dt.hour + ss.dt.minute / 60 + ss.dt.second / 3600).to_numpy(np.float32)

    # --- hourly temp grid (n, 24) -------------------------------------
    T_model = diurnal_temp(
        Tn=df["TMIN"].to_numpy(np.float32),
        Tx=df["TMAX"].to_numpy(np.float32),
        Tp=np.roll(df["TMIN"].to_numpy(np.float32), -1),  # next‑day Tmin
        Hn=Hn,
        Ho=Ho,
    )

    # --- hourly RH grid (Buck, vectorised) ----------------------------
    es = 0.61121 * np.exp((18.678 - (T_model / 234.5)) * T_model / (257.14 + T_model))
    tdew = df["TDEW"].to_numpy(np.float32)
    e = 0.61121 * np.exp((18.678 - (tdew / 234.5)) * tdew / (257.14 + tdew))
    hr = (e[:, None] / es) * 100.0  # shape (n, 24)

    # ------------------------------------------------------------------
    # 13:00 today → 12:00 next day   →   build (n‑1, 24) windows
    # ------------------------------------------------------------------
    T_win = np.hstack([T_model[:-1, 12:], T_model[1:, :12]])  # (n‑1, 24)
    hr_win = np.hstack([hr[:-1, 12:], hr[1:, :12]])

    above = hr_win >= rh_thresh
    hrs_abv = above.sum(axis=1).astype(np.int16)

    # Average temperature when RH ≥ threshold (else mean of 24 h)
    tmean_abv = np.where(
        hrs_abv > 0,
        (T_win * above).sum(axis=1) / hrs_abv.clip(min=1),
        T_win.mean(axis=1),
    ).astype(np.float32)

    # ------------------------------------------------------------------
    # Assemble output – drop first TWO & last TWO calendar days
    # ------------------------------------------------------------------
    out = pd.DataFrame(
        {
            "date": df["FECHA"].values[2:-2],  # day i (start at 13 h)
            "hr_ge_thresh": hrs_abv[1:-2],
            "tmean_ge_thresh": tmean_abv[1:-2],
            "rain_mm": df["PP"].values[2:-2],
            "tavg_C": df["tavg_C"].values[2:-2],
        }
    ).reset_index(drop=True)
    return out


@nb.njit(cache=True)
def _simcast_kernel(bu, fu, vt_idx, min_day, forced_day, abu, afu, app):
    """
    In-place scan computing ABU/AFU/APP and
    resetting BU & FU after an application.
    """
    n = bu.size
    bua = fua = 0
    days = 0
    app_ctr = 0
    first = True

    for k in range(n):
        days += 1
        bua += bu[k]
        fua += fu[k]

        forced = k == forced_day
        cutoff_bu = bua >= BU_THRESH[vt_idx]
        cutoff_fu = fua > FU_THRESH[vt_idx]
        decision = (cutoff_bu or cutoff_fu) and days > min_day and (not first)

        if forced or decision:
            app_ctr += 1
            days = 0
            abu[k] = bua
            afu[k] = fua
            bua = fua = 0
            fu[k] = 0  # reset today’s FU
            first = False

        app[k] = app_ctr


def simcast_model(
    df_in: pd.DataFrame, vt: str, *, min_day: int = 5, forced_day: int = 25
) -> pd.DataFrame:
    """
    Vector-accelerated SIMCAST daily model.
    """
    vt_idx = VT_INDEX[vt]
    df = df_in.copy().reset_index(drop=True)

    n = len(df)
    bu = np.empty(n, dtype=np.int16)
    fu = np.empty(n, dtype=np.int16)

    # ------------------------------------------------------------
    # 3.1  compute BU / FU in one shot
    calc_bu_vec(
        df["hr_ge_thresh"].to_numpy(np.float32),
        df["tmean_ge_thresh"].to_numpy(np.float32),
        vt_idx,
        bu,
    )

    dsa = np.arange(1, n + 1, dtype=np.int16)
    calc_fu_vec(df["rain_mm"].to_numpy(np.float32), dsa, fu)

    # ------------------------------------------------------------
    # 3.2  scan for applications (Numba kernel)
    abu = np.zeros(n, dtype=np.int8)
    afu = np.zeros(n, dtype=np.int8)
    app = np.zeros(n, dtype=np.int16)

    _simcast_kernel(
        bu,
        fu,
        vt_idx,
        min_day=min_day,
        forced_day=forced_day,
        abu=abu,
        afu=afu,
        app=app,
    )

    return df.assign(BU=bu, FU=fu, ABU=abu, AFU=afu, APP=app)
