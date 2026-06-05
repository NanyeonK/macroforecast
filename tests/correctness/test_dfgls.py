"""Correctness tests for dfgls_test (urca::ur.ers / arch.unitroot.DFGLS)."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("arch")

import macroforecast as mf


def test_rejects_unit_root_for_stationary():
    rng = np.random.default_rng(0)
    n = 300
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.2 * x[t - 1] + rng.standard_normal()
    res = mf.data_analysis.dfgls_test(x, trend="c")
    assert res["test"] == "dfgls"
    assert res["reject_unit_root"] is True
    assert res["statistic"] < res["critical_values"]["5%"]


def test_does_not_reject_for_random_walk():
    # Average over seeds: DF-GLS should mostly fail to reject a true unit root.
    rejects = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        y = np.cumsum(rng.standard_normal(300))
        if mf.data_analysis.dfgls_test(y, trend="c")["reject_unit_root"]:
            rejects += 1
    # nominal 5% size -> well under half should reject
    assert rejects <= 5


def test_structure_and_trend_option():
    rng = np.random.default_rng(2)
    y = np.cumsum(rng.standard_normal(200)) + np.arange(200) * 0.05
    res = mf.data_analysis.dfgls_test(y, trend="ct")
    assert set(res) >= {"statistic", "p_value", "used_lag", "critical_values", "trend"}
    assert res["trend"] == "ct"
    with pytest.raises(ValueError):
        mf.data_analysis.dfgls_test(y, trend="x")
