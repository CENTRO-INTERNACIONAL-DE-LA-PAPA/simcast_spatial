import numpy as np
import pandas as pd
import pytest

from el_nino_analysis import classify_years, histogram, kl_divergence


def _events():
    return pd.DataFrame({
        "start": pd.PeriodIndex(["1982-09", "2017-01"], freq="M"),
        "end": pd.PeriodIndex(["1983-08", "2017-05"], freq="M"),
        "magnitude": ["extraordinario", "moderado"],
    })


def test_classify_years():
    t = classify_years(_events(), [1981, 1982, 1983, 2016, 2017], min_overlap_months=3).set_index("year")
    assert not t.loc[1981, "is_nino"]
    # 1982: Sierra Alta season Nov82-Apr83 fully inside the event
    assert t.loc[1982, "is_nino"] and t.loc[1982, "magnitude"] == "extraordinario"
    assert t.loc[1982, "overlap_Sierra Alta"] == 6
    # 1983: Sierra Media Apr-Jul 83 inside the event
    assert t.loc[1983, "is_nino"]
    # 2017 event Jan-May: 4 months of Sierra Alta 2016 (Jan-Apr 17), only 2 of Sierra Media 2017
    assert t.loc[2016, "is_nino"] and t.loc[2016, "magnitude"] == "moderado"
    assert not t.loc[2017, "is_nino"]
    assert t.loc[2017, "overlap_Sierra Media"] == 2


def test_kl_basic():
    p = histogram(np.array([0, 1, 1, 2]), 3)
    assert kl_divergence(p, p) == pytest.approx(0.0)
    q = histogram(np.array([0, 0, 0, 2]), 3)
    assert kl_divergence(p, q) > 0
    # no smoothing: matches the textbook formula on strictly positive histograms
    a, b = np.array([2.0, 1.0, 1.0]), np.array([1.0, 1.0, 2.0])
    pa, pb = a / a.sum(), b / b.sum()
    assert kl_divergence(a, b, alpha=0.0) == pytest.approx(np.sum(pa * np.log(pa / pb)))


def test_histogram_bin_width():
    h = histogram(np.array([0.0, 9.9, 10.0, 25.0]), 3, bin_width=10.0)
    assert h.tolist() == [2.0, 1.0, 1.0]
