"""Correctness tests for variance_ratio (Lo-MacKinlay; arch VarianceRatio)."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("arch")

import macroforecast as mf


def test_random_walk_not_rejected():
    rng = np.random.default_rng(0)
    rw = np.cumsum(rng.standard_normal(600))
    res = mf.data_analysis.variance_ratio(rw, lags=4)
    assert res["test"] == "variance_ratio"
    assert res["reject_random_walk"] is False
    assert abs(res["variance_ratio"] - 1.0) < 0.2


def test_mean_reversion_rejects_random_walk():
    rng = np.random.default_rng(1)
    n = 600
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = -0.4 * e[t - 1] + rng.standard_normal()
    res = mf.data_analysis.variance_ratio(np.cumsum(e), lags=4)
    assert res["reject_random_walk"] is True
    assert res["variance_ratio"] < 1.0


def test_matches_arch():
    from arch.unitroot import VarianceRatio
    rng = np.random.default_rng(5)
    x = np.cumsum(rng.standard_normal(500))
    res = mf.data_analysis.variance_ratio(x, lags=8, robust=True)
    ref = VarianceRatio(x, lags=8, robust=True)
    np.testing.assert_allclose(res["variance_ratio"], ref.vr, rtol=1e-8)
    np.testing.assert_allclose(res["p_value"], ref.pvalue, rtol=1e-6)


def test_invalid_lags():
    with pytest.raises(ValueError):
        mf.data_analysis.variance_ratio(np.arange(50.0), lags=1)
