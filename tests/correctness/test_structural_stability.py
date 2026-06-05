"""Correctness tests for structural_stability (OLS-CUSUM; strucchange::efp)."""
import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def test_stable_series_not_rejected():
    rng = np.random.default_rng(0)
    n = 300
    x = rng.standard_normal(n)
    y = 1.0 + 0.5 * x + rng.standard_normal(n)
    res = mf.data_analysis.structural_stability(y, x)
    assert res["test"] == "ols_cusum"
    assert res["reject_stability"] is False
    assert res["p_value"] > 0.05


def test_intercept_break_detected():
    rng = np.random.default_rng(1)
    n = 300
    x = rng.standard_normal(n)
    y = 1.0 + 0.5 * x + rng.standard_normal(n)
    y[150:] += 3.0  # structural break in the intercept
    res = mf.data_analysis.structural_stability(y, x)
    assert res["reject_stability"] is True
    assert 100 <= res["break_index"] <= 200  # break located near the middle


def test_critical_values_match_known():
    res = mf.data_analysis.structural_stability(
        np.arange(100.0) + np.random.default_rng(0).standard_normal(100),
        np.arange(100.0),
    )
    # known strucchange OLS-CUSUM sup|BB| critical values
    assert abs(res["critical_values"]["5%"] - 1.3581) < 1e-2
    assert abs(res["critical_values"]["1%"] - 1.6276) < 1e-2


def test_break_label_from_datetime_index():
    rng = np.random.default_rng(2)
    n = 200
    idx = pd.date_range("1990-01-31", periods=n, freq="ME", name="date")
    x = rng.standard_normal(n)
    y = pd.Series(0.5 * x + rng.standard_normal(n), index=idx)
    y.iloc[120:] += 4.0
    res = mf.data_analysis.structural_stability(pd.DataFrame({"y": y, "x": x}, index=idx))
    assert res["break_label"] is not None
    assert res["reject_stability"] is True


def test_mean_only_stability():
    # no regressors -> tests stability of the mean
    rng = np.random.default_rng(3)
    s = pd.Series(rng.standard_normal(200))
    res = mf.data_analysis.structural_stability(pd.DataFrame({"y": s}))
    assert res["n_coef"] == 1
    assert res["reject_stability"] is False
