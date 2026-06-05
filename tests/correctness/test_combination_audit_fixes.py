"""Regression tests for independent-audit findings on the combination kernel."""
import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.models import _weight_solvers as ws


def test_constrained_ls_recovers_binding_optimum_not_equal_fallback():
    # Construct a problem whose unconstrained optimum has negative weights, so the
    # simplex constraint binds (the case SLSQP flags success=False on).
    rng = np.random.default_rng(0)
    n, k = 200, 6
    F = rng.standard_normal((n, k))
    true_w = np.array([0.8, -0.3, 0.25, 0.0, 0.15, 0.1])  # has negatives
    y = F @ true_w + rng.standard_normal(n) * 0.01
    obj = lambda v: float(np.sum((y - F @ v) ** 2))
    equal = np.full(k, 1.0 / k)
    # stress many binding problems: none should silently fall back to equal weights
    fallbacks = 0
    for seed in range(40):
        r = np.random.default_rng(100 + seed)
        Fi = r.standard_normal((n, k))
        yi = Fi @ true_w + r.standard_normal(n) * 0.01
        wi = ws.constrained_ls_weights(Fi, yi)
        assert np.all(wi >= -1e-9) and abs(wi.sum() - 1.0) < 1e-6
        if float(np.sum((yi - Fi @ wi) ** 2)) > float(np.sum((yi - Fi @ equal) ** 2)) * 0.95:
            fallbacks += 1
    assert fallbacks == 0


def test_min_variance_uncentered_downweights_biased_model():
    # model 0 unbiased unit variance; model 1 large constant bias -> should be downweighted
    rng = np.random.default_rng(1)
    n = 500
    e = np.column_stack([rng.standard_normal(n), rng.standard_normal(n) * 0.5 + 3.0])
    w = ws.min_variance_weights(e)
    assert w[0] > w[1]            # unbiased model preferred
    assert abs(w.sum() - 1.0) < 1e-9


def test_eigenvector_degenerate_falls_back_not_blowup():
    rng = np.random.default_rng(2)
    n = 100
    base = rng.standard_normal(n)
    # near-collinear errors -> smallest-eigenvalue eigenvector near-orthogonal to ones
    e = np.column_stack([base, base + 1e-9 * rng.standard_normal(n), base - 1e-9 * rng.standard_normal(n)])
    w = ws.eigenvector_weights(e)
    assert np.all(np.isfinite(w))
    assert np.max(np.abs(w)) < 100.0   # no 1e10 blowup


def test_regression_constrained_sums_to_one_even_collinear():
    rng = np.random.default_rng(3)
    n = 200
    a = rng.standard_normal(n)
    F = np.column_stack([a, 2.0 * a, a + 1e-6 * rng.standard_normal(n)])  # collinear
    y = a + rng.standard_normal(n) * 0.1
    w, b = ws.regression_weights(F, y, sum_to_one=True)
    assert abs(w.sum() - 1.0) < 1e-8


def test_log_pool_rejects_nonpositive_sigma():
    means = pd.DataFrame({"a": [1.0], "b": [4.0]})
    sds = pd.DataFrame({"a": [0.0], "b": [2.0]})
    with pytest.raises(ValueError):
        mf.forecasting.combine_log_pool(means, sds)
