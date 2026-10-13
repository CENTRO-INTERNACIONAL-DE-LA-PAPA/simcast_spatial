import numpy as np
import pytest

from el_nino_regional import fdr_mask, js_bits, permutation_pvalues, wasserstein_1, zone_of


def test_js_bits_bounds():
    a = np.array([10.0, 0.0])
    b = np.array([0.0, 10.0])
    assert js_bits(a, a) == pytest.approx(0.0)
    assert js_bits(a, b) == pytest.approx(1.0)  # disjoint supports -> 1 bit
    c = np.array([5.0, 5.0])
    assert 0 < js_bits(a, c) < 1
    assert js_bits(a, c) == pytest.approx(js_bits(c, a))  # symmetric


def test_wasserstein_shift():
    p = np.bincount([2, 3, 4], minlength=10).astype(float)
    q = np.bincount([4, 5, 6], minlength=10).astype(float)
    assert wasserstein_1(p, q) == pytest.approx(2.0)  # shifted by 2 units
    assert wasserstein_1(p, p) == 0.0


def test_fdr_mask():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    m = fdr_mask(p, alpha=0.05)
    assert m.tolist() == [True, True] + [False] * 8


def test_permutation_detects_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, (200, 21))
    grp_null = rng.normal(0, 1, (200, 4))
    grp_shift = rng.normal(3, 1, (200, 4))
    x = np.hstack([ref, grp_null, grp_shift])
    p_null = permutation_pvalues(x, np.arange(21, 25), np.arange(21), n_perm=500)
    p_shift = permutation_pvalues(x, np.arange(25, 29), np.arange(21), n_perm=500)
    assert np.mean(p_null < 0.05) < 0.15
    assert np.mean(p_shift < 0.05) > 0.95


def test_zones():
    z = zone_of(np.array([-5.0, -9.0, -12.0, -15.0, -17.0]))
    assert z.tolist() == ["Norte", "Norte", "Centro", "Centro", "Sur"]  # Norte >= -9, Sur < -15


def test_field_significance():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, (300, 1))  # spatially correlated noise
    ref = base + rng.normal(0, 1, (300, 21))
    shift = base + rng.normal(0, 1, (300, 4)) + 2.0
    null = base + rng.normal(0, 1, (300, 4))
    x1 = np.hstack([ref, shift])
    x0 = np.hstack([ref, null])
    p1, f1 = permutation_pvalues(x1, np.arange(21, 25), np.arange(21), return_field=True)
    p0, f0 = permutation_pvalues(x0, np.arange(21, 25), np.arange(21), return_field=True)
    assert np.mean(f1 >= np.mean(p1 <= 0.05)) < 0.01  # shifted field: significant
    assert np.mean(f0 >= np.mean(p0 <= 0.05)) > 0.05  # null field: not significant
