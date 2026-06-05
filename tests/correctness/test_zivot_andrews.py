"""Correctness tests for zivot_andrews_test (urca::ur.za)."""
import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def test_detects_break_and_rejects_for_broken_stationary():
    rng = np.random.default_rng(1)
    n = 200
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.4 * x[t - 1] + rng.standard_normal()
    x[100:] += 6.0  # level break at t=100
    res = mf.data_analysis.zivot_andrews_test(x, regression="c")
    assert res["test"] == "zivot_andrews"
    assert res["reject_unit_root"] is True
    # break located near the true breakpoint
    assert 80 <= res["break_index"] <= 120


def test_break_label_from_datetime_index():
    rng = np.random.default_rng(3)
    n = 160
    x = np.cumsum(rng.standard_normal(n))
    idx = pd.date_range("1990-01-01", periods=n, freq="MS", name="date")
    s = pd.Series(x, index=idx)
    res = mf.data_analysis.zivot_andrews_test(s, regression="ct")
    assert res["break_label"] is not None
    assert set(res) >= {"statistic", "p_value", "critical_values", "break_index"}


def test_invalid_regression():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal(100))
    with pytest.raises(ValueError):
        mf.data_analysis.zivot_andrews_test(x, regression="z")
