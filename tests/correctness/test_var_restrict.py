"""Correctness tests for var_restrict (vars::restrict sequential elimination)."""
import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def _var_panel(n=300, seed=0):
    # y depends on its own lag and x's lag; x is an independent AR(1).
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + rng.standard_normal()
        y[t] = 0.4 * y[t - 1] + 0.6 * x[t - 1] + rng.standard_normal()
    idx = pd.date_range("1990-01-31", periods=n, freq="ME", name="date")
    return pd.DataFrame({"y": y, "x": x}, index=idx)


def test_structure():
    res = mf.models.var_restrict(_var_panel(), n_lag=1, threshold=2.0)
    assert res["n_vars"] == 2
    assert len(res["equations"]) == 2
    assert len(res["restriction_matrix"]) == 2
    # restriction matrix entries are 0/1
    flat = [v for row in res["restriction_matrix"] for v in row]
    assert set(flat) <= {0, 1}


def test_retains_true_predictors():
    res = mf.models.var_restrict(_var_panel(), n_lag=1, threshold=2.0)
    y_eq = next(e for e in res["equations"] if e["equation"] == "y")
    # y depends on y.l1 and x.l1 -> both should survive elimination
    assert "y.l1" in y_eq["retained"]
    assert "x.l1" in y_eq["retained"]


def test_higher_threshold_eliminates_more():
    panel = _var_panel(seed=1)
    low = mf.models.var_restrict(panel, n_lag=2, threshold=1.0)
    high = mf.models.var_restrict(panel, n_lag=2, threshold=5.0)
    assert high["n_restricted"] >= low["n_restricted"]


def test_validation():
    with pytest.raises(ValueError):
        mf.models.var_restrict(pd.DataFrame({"y": [1.0, 2.0, 3.0]}))  # single column
