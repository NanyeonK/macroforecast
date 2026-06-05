"""Regression: risk_forecast honours the skew-t / GED innovation tail (parity FIX)."""
from __future__ import annotations
import numpy as np, pandas as pd
import macroforecast as mf
import pytest
pytest.importorskip("arch")

def _returns(n=800, seed=0):
    rng = np.random.default_rng(seed)
    r = np.zeros(n); h = 1.0
    for t in range(1, n):
        h = 0.05 + 0.08 * r[t-1]**2 + 0.9 * h
        r[t] = np.sqrt(h) * rng.standard_t(df=5)
    return pd.Series(r, index=pd.date_range("2000-01-31", periods=n, freq="ME"))

def _arch_tail(fit, alpha):
    d = fit.estimator._fitted.model.distribution
    n = int(d.num_params)
    shape = fit.estimator._fitted.params.to_numpy()[-n:] if n else None
    return float(np.asarray(d.ppf(np.array([alpha]), shape)).reshape(-1)[0])

def test_skewt_tail_uses_arch_ppf_not_symmetric_t():
    fit = mf.models.garch11(_returns(), dist="skewt")
    out = mf.models.risk_forecast(fit, alpha=0.05, horizon=1)
    assert out["distribution"].startswith("arch:") and "Skew" in out["distribution"]
    # standardized VaR quantile must equal arch's skew-t ppf
    z = (float(out["var"][0]) - out["mean"]) / float(out["sigma"][0])
    np.testing.assert_allclose(z, _arch_tail(fit, 0.05), rtol=1e-6)

def test_ged_tail_uses_arch_ppf_not_normal():
    fit = mf.models.garch11(_returns(seed=1), dist="ged")
    out = mf.models.risk_forecast(fit, alpha=0.01, horizon=1)
    assert out["distribution"].startswith("arch:") and "Generalized" in out["distribution"]
    z = (float(out["var"][0]) - out["mean"]) / float(out["sigma"][0])
    np.testing.assert_allclose(z, _arch_tail(fit, 0.01), rtol=1e-6)
    # GED tail differs from the (wrong) normal fallback
    from scipy.stats import norm
    assert not np.isclose(z, norm.ppf(0.01), rtol=1e-3)
