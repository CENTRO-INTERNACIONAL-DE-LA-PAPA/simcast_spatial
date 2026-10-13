import numpy as np
import pandas as pd
import pytest

from el_nino_icen import (block_permutations, category, load_icen, season_icen, season_label,
                          slopes_with_permutation, zone_of)


def test_season_icen_uses_each_region_months():
    idx = pd.period_range("1997-01", "1999-12", freq="M")
    icen = pd.Series(np.arange(len(idx), dtype=float), index=idx)  # value = months since Jan 1997
    s = season_icen(icen, [1997, 1998])
    assert s.loc[1997, "Sierra Media"] == pytest.approx(np.mean([3, 4, 5, 6]))  # Apr-Jul 1997
    assert s.loc[1997, "Costa y Valle Interandino"] == pytest.approx(np.mean(range(6, 12)))
    assert s.loc[1998, "Sierra Alta"] == pytest.approx(np.mean(range(22, 28)))  # Nov 1998 - Apr 1999
    assert season_label("Sierra Alta", 1998) == "1998/99"


def test_official_icen_1998_sierra_alta_is_neutral():
    icen = load_icen("reference/icen_enfen2024.txt")
    s = season_icen(icen, [1997, 1998])
    assert s.loc[1997, "Sierra Alta"] > 3.0
    assert category(s.loc[1998, "Sierra Alta"]) == "neutra"


def test_block_permutations_are_permutations_and_keep_pairs():
    p = block_permutations(39, 200, seed=1)
    assert all(sorted(r) == list(range(39)) for r in p)
    # most consecutive pairs stay adjacent
    adj = np.mean([np.any(np.abs(np.diff(r)) == 1) for r in p])
    assert adj == 1.0


def test_slope_matches_polyfit_and_detects_signal():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 39)
    y_sig = 2.0 * x[None, :] + rng.normal(0, 1, (300, 39))
    y_null = rng.normal(0, 1, (300, 39))
    perms = block_permutations(39, 1000)
    b, p, f = slopes_with_permutation(y_sig, x, perms)
    assert b[0] == pytest.approx(np.polyfit(x, y_sig[0], 1)[0], rel=1e-4)
    assert np.mean(p < 0.05) > 0.95
    b0, p0, f0 = slopes_with_permutation(y_null, x, perms)
    share0 = np.mean(p0 < 0.05)
    assert share0 < 0.15
    assert np.mean(f0 / 300 >= share0) > 0.05  # null field not significant


def test_zone_and_category():
    assert zone_of(np.array([-5.0, -9.0, -15.0, -16.0])).tolist() == ["Norte", "Norte", "Centro", "Sur"]
    assert category(-1.0) == "fría" and category(0.0) == "neutra" and category(3.9) == "cálida extraordinaria"
