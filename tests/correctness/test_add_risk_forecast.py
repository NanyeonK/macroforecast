"""ADD: VaR / ES forecast from a fitted volatility model."""
from __future__ import annotations
import numpy as np, pandas as pd
import macroforecast as mf
import pytest
pytest.importorskip("arch")
from scipy.stats import norm


def _returns(n=800, seed=0):
    rng = np.random.default_rng(seed)
    r = np.zeros(n); h = 1.0
    for t in range(1, n):
        h = 0.05 + 0.08 * r[t-1]**2 + 0.9 * h
        r[t] = np.sqrt(h) * rng.normal()
    return pd.Series(r, index=pd.date_range("2000-01-31", periods=n, freq="ME"))


def test_risk_forecast_normal_matches_closed_form():
    fit = mf.models.garch11(_returns(), dist="normal")
    out = mf.models.risk_forecast(fit, alpha=0.05, horizon=1)
    assert out["distribution"] == "normal"
    sigma = float(out["sigma"][0]); mu = out["mean"]
    var = float(out["var"][0]); es = float(out["es"][0])
    # closed-form normal VaR/ES
    np.testing.assert_allclose(var, mu + sigma * norm.ppf(0.05), rtol=1e-9)
    np.testing.assert_allclose(es, mu - sigma * norm.pdf(norm.ppf(0.05)) / 0.05, rtol=1e-9)
    assert es < var < 0                                   # ES strictly more extreme than VaR
    assert abs((es - mu) / (var - mu) - (norm.pdf(norm.ppf(0.05))/0.05) / (-norm.ppf(0.05))) < 1e-9


def test_value_at_risk_and_es_helpers_multistep():
    fit = mf.models.garch11(_returns(seed=1))
    v = mf.models.value_at_risk(fit, alpha=0.01, horizon=5)
    e = mf.models.expected_shortfall(fit, alpha=0.01, horizon=5)
    assert v.shape == (5,) and e.shape == (5,)
    assert np.all(e <= v)                                  # ES no less extreme than VaR
