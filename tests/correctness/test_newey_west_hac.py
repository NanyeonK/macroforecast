"""Correctness tests for the public newey_west HAC estimator (sandwich::NeweyWest)."""
import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def _ar1_errors(n, rho, rng):
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = rho * e[t - 1] + rng.standard_normal()
    return e


def test_matches_statsmodels_hac():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(42)
    n = 200
    x1, x2 = rng.standard_normal(n), rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 - 1.5 * x2 + _ar1_errors(n, 0.6, rng)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    L = 4
    res = mf.data_analysis.newey_west(X, y, lags=L, small_sample=False)
    sm_res = sm.OLS(y, sm.add_constant(X.values)).fit(
        cov_type="HAC", cov_kwds={"maxlags": L, "use_correction": False}
    )
    np.testing.assert_allclose(res["estimate"], sm_res.params, rtol=1e-10)
    np.testing.assert_allclose(res["std_error"], sm_res.bse, rtol=1e-8)


def test_auto_bandwidth_newey_west_rule():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.standard_normal(n)
    y = x + rng.standard_normal(n)
    res = mf.data_analysis.newey_west(pd.DataFrame({"x": x}), y)
    assert res["lags"] == int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def test_hac_se_exceeds_ols_under_positive_autocorrelation():
    # With positively autocorrelated errors HAC SEs should exceed the naive
    # OLS SE that ignores serial correlation (lags=0 reproduces White HC0).
    rng = np.random.default_rng(3)
    n = 300
    x = rng.standard_normal(n)
    y = 1.0 + 0.5 * x + _ar1_errors(n, 0.7, rng)
    X = pd.DataFrame({"x": x})
    hac = mf.data_analysis.newey_west(X, y, lags=8)
    white = mf.data_analysis.newey_west(X, y, lags=0)
    # slope is index 1 (intercept is 0)
    assert hac["std_error"][1] > white["std_error"][1]


def test_target_plus_regressors_single_frame():
    rng = np.random.default_rng(5)
    n = 150
    df = pd.DataFrame(
        {"y": rng.standard_normal(n), "x1": rng.standard_normal(n), "x2": rng.standard_normal(n)}
    )
    res = mf.data_analysis.newey_west(df)
    assert res["names"][0] == "(intercept)"
    assert res["n_coef"] == 3
    assert len(res["coefficients"]) == 3


def test_requires_more_obs_than_coef():
    df = pd.DataFrame({"y": [1.0, 2.0], "x": [0.1, 0.2]})
    with pytest.raises(ValueError):
        mf.data_analysis.newey_west(df)
