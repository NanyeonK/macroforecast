"""ADD: var_impulse_response with bootstrap confidence bands."""
from __future__ import annotations
import numpy as np, pandas as pd
import macroforecast as mf

def test_var_irf_table_bands_and_directionality():
    rng = np.random.default_rng(0); n = 400
    x = np.zeros(n); y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.5 * x[t-1] + rng.normal()
        y[t] = 0.4 * y[t-1] + 0.4 * x[t-1] + rng.normal()   # shock to x raises y
    df = pd.DataFrame({"x": x, "y": y})
    tab = mf.interpretation.var_impulse_response(df, n_lag=2, periods=6, repl=200, seed=1)
    assert set(tab.columns) == {"horizon", "impulse", "response", "irf", "lower", "upper"}
    assert len(tab) == 7 * 2 * 2                      # (periods+1) * k * k
    # bootstrap percentile bands are well-ordered (lower <= upper)
    assert (tab["lower"] <= tab["upper"] + 1e-9).all()
    # response of y to an x impulse is positive at short horizon
    xy = tab[(tab["impulse"] == "x") & (tab["response"] == "y") & (tab["horizon"] == 1)]
    assert float(xy["irf"].iloc[0]) > 0
    assert tab.attrs["macroforecast_metadata"]["orthogonalized"] is True
