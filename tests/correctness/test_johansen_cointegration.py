"""Correctness tests for johansen_cointegration (ca.jo analogue)."""
import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def _rw(n, rng, scale=1.0):
    return np.cumsum(rng.standard_normal(n) * scale)


def test_single_cointegration_relation():
    # x is a random walk; y = x + stationary noise -> exactly one
    # cointegrating relation; z is an independent random walk.
    rng = np.random.default_rng(0)
    n = 400
    x = _rw(n, rng)
    y = x + rng.standard_normal(n) * 0.5
    z = _rw(n, rng)
    idx = pd.date_range("1990-01-01", periods=n, freq="MS", name="date")
    df = pd.DataFrame({"x": x, "y": y, "z": z}, index=idx)
    res = mf.data_analysis.johansen_cointegration(df, k_ar_diff=1)
    assert res["n_vars"] == 3
    assert res["cointegration_rank"]["trace"] == 1


def test_no_cointegration_independent_walks():
    rng = np.random.default_rng(7)
    n = 400
    idx = pd.date_range("1990-01-01", periods=n, freq="MS", name="date")
    df = pd.DataFrame(
        {name: _rw(n, rng) for name in ("a", "b", "c")}, index=idx
    )
    res = mf.data_analysis.johansen_cointegration(df, k_ar_diff=1)
    assert res["cointegration_rank"]["trace"] == 0


def test_structure_and_keys():
    rng = np.random.default_rng(3)
    n = 300
    x = _rw(n, rng)
    df = pd.DataFrame({"x": x, "y": x + rng.standard_normal(n) * 0.4})
    res = mf.data_analysis.johansen_cointegration(df)
    assert set(res) >= {
        "n_vars", "names", "trace", "max_eigen", "eigenvalues",
        "cointegration_rank", "cointegrating_vectors",
    }
    assert len(res["trace"]) == res["n_vars"]
    assert len(res["max_eigen"]) == res["n_vars"]
    for row in res["trace"]:
        assert {"rank_null", "statistic", "reject"} <= set(row)
