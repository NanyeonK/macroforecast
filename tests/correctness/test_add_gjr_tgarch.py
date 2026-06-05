"""ADD: gjr_garch and tgarch asymmetric volatility models."""
from __future__ import annotations
import numpy as np, pandas as pd
import macroforecast as mf
import pytest

pytest.importorskip("arch")

def _returns(n=600, seed=0):
    rng = np.random.default_rng(seed)
    r = np.zeros(n); h = 1.0
    for t in range(1, n):
        shock = rng.normal()
        # leverage: negative shocks raise next variance more
        lev = 0.1 if r[t-1] < 0 else 0.0
        h = 0.05 + (0.05 + lev) * r[t-1]**2 + 0.9 * h
        r[t] = np.sqrt(h) * shock
    return pd.Series(r, index=pd.date_range("2000-01-31", periods=n, freq="ME"))

def test_gjr_and_tgarch_fit_and_forecast():
    r = _returns()
    for fn, name in ((mf.models.gjr_garch, "gjr_garch"), (mf.models.tgarch, "tgarch")):
        fit = fn(r, o=1)
        assert fit.metadata["o"] == 1 and fit.metadata["model"] == name if "model" in fit.metadata else True
        var = fit.estimator.predict_variance(horizon=3)
        assert var.shape == (3,) and np.all(np.isfinite(var)) and np.all(var > 0)
