import json

import numpy as np
import pandas as pd
import pytest

from trend_significance import (
    aggregate_season_file,
    build_yearly_table,
    ols_trend_by_id,
)


def test_ols_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(0)
    rows = []
    truth = {}
    for i in range(25):
        years = np.arange(1981, 2021)
        keep = rng.uniform(size=len(years)) > 0.15  # random gaps
        y = 3 + 0.05 * (years - 2000) * rng.normal(1, 0.5) + rng.normal(0, 1, len(years))
        y = y[keep]
        x = years[keep]
        rows.append(pd.DataFrame({"ID": i, "year": x, "APP": y}))
        X = sm.add_constant(x.astype(float))
        m = sm.OLS(y, X).fit()
        ci = m.conf_int()[1]
        truth[i] = (m.params[1], m.pvalues[1], m.bse[1], ci[0], ci[1], m.rsquared, len(x))
    yearly = pd.concat(rows)

    out = ols_trend_by_id(yearly, "APP", min_years=10).set_index("ID")
    for i, (slope, p, se, lo, hi, r2, n) in truth.items():
        r = out.loc[i]
        assert np.isclose(r["slope_app"], slope, rtol=1e-9)
        assert np.isclose(r["p_value_app"], p, rtol=1e-8, atol=1e-12)
        assert np.isclose(r["std_err_app"], se, rtol=1e-9)
        assert np.isclose(r["conf_int_lower_app"], lo, rtol=1e-9)
        assert np.isclose(r["conf_int_upper_app"], hi, rtol=1e-9)
        assert np.isclose(r["r_squared_app"], r2, rtol=1e-9)
        assert r["n_years_app"] == n


def test_constant_series_is_no_trend():
    yearly = pd.DataFrame({"ID": [1] * 20, "year": range(1990, 2010), "ABU": 0.0})
    out = ols_trend_by_id(yearly, "ABU", min_years=10).set_index("ID")
    assert out.loc[1, "slope_abu"] == 0.0
    assert out.loc[1, "p_value_abu"] == 1.0
    assert np.isnan(out.loc[1, "r_squared_abu"])


def test_ols_min_years_and_range():
    yearly = pd.DataFrame(
        {"ID": [1] * 5 + [2] * 20, "year": list(range(2000, 2005)) + list(range(1990, 2010)),
         "ABU": list(range(5)) + list(range(20))}
    )
    out = ols_trend_by_id(yearly, "ABU", min_years=10).set_index("ID")
    assert np.isnan(out.loc[1, "slope_abu"])
    assert np.isclose(out.loc[2, "slope_abu"], 1.0)
    out = ols_trend_by_id(yearly, "ABU", min_years=5, year_range=(1990, 1999)).set_index("ID")
    assert 1 not in out.index  # filtered out entirely
    assert out.loc[2, "n_years_abu"] == 10


def _write_season(root, year, period, region, ids, dates, truncated=False, app_base=0):
    d = root / str(year) / period / region
    d.mkdir(parents=True)
    frames = []
    for k, i in enumerate(ids):
        n = len(dates)
        frames.append(
            pd.DataFrame(
                {"ID": i, "date": dates, "BU": np.full(n, k + 1), "FU": np.full(n, 2),
                 "APP": np.minimum(np.arange(n) // 10, 3) + app_base}
            )
        )
    pd.concat(frames).to_parquet(d / "simcast_results.parquet", index=False)
    with open(d / "coverage.json", "w") as fh:
        json.dump({"truncated": truncated}, fh)


def test_season_vs_calendar_aggregation(tmp_path):
    root = tmp_path / "res"
    # cultivo1 spans Nov..Apr: season 2000 covers 2000-11-03 .. 2001-04-28
    dates = pd.date_range("2000-11-03", "2001-04-28")
    _write_season(root, 2000, "cultivo1", "Sierra Alta", [7, 8], dates)
    _write_season(root, 2001, "cultivo1", "Sierra Alta", [7, 8],
                  pd.date_range("2001-11-03", "2002-04-28"), truncated=True)

    season = aggregate_season_file(str(root / "2000/cultivo1/Sierra Alta/simcast_results.parquet"))
    assert season.year.tolist() == [2000, 2000]
    assert season.set_index("ID").loc[7, "ABU"] == len(dates) * 1
    assert season.set_index("ID").loc[8, "ABU"] == len(dates) * 2
    assert (season.APP == 3).all()
    assert (season.n_days == len(dates)).all()

    cal = aggregate_season_file(
        str(root / "2000/cultivo1/Sierra Alta/simcast_results.parquet"), aggregate_by="calendar_year"
    )
    assert sorted(cal.year.unique()) == [2000, 2001]  # split across two years

    yearly = build_yearly_table(root, workers=1)
    assert yearly.year.unique().tolist() == [2000]  # truncated 2001 excluded
    yearly = build_yearly_table(root, workers=1, exclude_truncated=False)
    assert sorted(yearly.year.unique()) == [2000, 2001]
