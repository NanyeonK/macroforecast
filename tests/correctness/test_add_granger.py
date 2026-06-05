"""ADD: granger_causality + instantaneous_causality."""
from __future__ import annotations
import numpy as np, pandas as pd
import macroforecast as mf

def _panel(n=400, seed=0):
    rng = np.random.default_rng(seed); x = np.zeros(n); y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.5 * x[t-1] + rng.normal()
        y[t] = 0.4 * y[t-1] + 0.35 * x[t-1] + rng.normal()   # x Granger-causes y, not vice versa
    return pd.DataFrame({"x": x, "y": y})

def test_granger_directionality():
    df = _panel()
    xy = mf.tests.granger_causality(df, caused="y", causing="x", n_lag=2)
    yx = mf.tests.granger_causality(df, caused="x", causing="y", n_lag=2)
    assert xy.decision is True and xy.p_value < 0.05     # x -> y
    assert yx.decision is False                          # y -/-> x
    assert xy.metadata["causing"] == ["x"] and xy.metadata["kind"] == "f"

def test_instantaneous_causality_runs():
    ic = mf.tests.instantaneous_causality(_panel(), caused="x", n_lag=2)
    assert ic.p_value is not None and ic.alternative == "instantaneous_causality"
