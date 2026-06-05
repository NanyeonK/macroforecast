"""Correctness tests for engle_granger cointegration test (statsmodels coint)."""
import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def test_detects_cointegration():
    rng = np.random.default_rng(0)
    n = 400
    x = np.cumsum(rng.standard_normal(n))
    y = 2.0 + 1.5 * x + rng.standard_normal(n) * 0.5
    res = mf.data_analysis.engle_granger(y, x)
    assert res["test"] == "engle_granger"
    assert res["cointegrated"] is True
    assert res["p_value"] < 0.05


def test_independent_random_walks_not_cointegrated():
    rng = np.random.default_rng(1)
    n = 400
    y = np.cumsum(rng.standard_normal(n))
    z = np.cumsum(rng.standard_normal(n))
    res = mf.data_analysis.engle_granger(y, z)
    assert res["cointegrated"] is False


def test_matches_statsmodels_coint():
    sm = pytest.importorskip("statsmodels.tsa.stattools")
    rng = np.random.default_rng(7)
    n = 300
    x = np.cumsum(rng.standard_normal(n))
    y = 1.0 + 0.8 * x + rng.standard_normal(n) * 0.4
    res = mf.data_analysis.engle_granger(y, x)
    stat, p, _ = sm.coint(y, x, trend="c", autolag="aic")
    np.testing.assert_allclose(res["statistic"], stat, rtol=1e-8)
    np.testing.assert_allclose(res["p_value"], p, rtol=1e-6)


def test_single_panel_and_coefficients():
    rng = np.random.default_rng(3)
    n = 300
    x = np.cumsum(rng.standard_normal(n))
    y = 5.0 + 2.0 * x + rng.standard_normal(n) * 0.3
    df = pd.DataFrame({"y": y, "x": x})
    res = mf.data_analysis.engle_granger(df)
    # cointegrating slope on x recovered near 2.0
    assert abs(res["cointegrating_coef"]["x"] - 2.0) < 0.1
    assert "(intercept)" in res["cointegrating_coef"]


def test_requires_regressor():
    with pytest.raises(ValueError):
        mf.data_analysis.engle_granger(pd.DataFrame({"y": [1.0, 2.0, 3.0]}))
