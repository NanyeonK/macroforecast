"""ADD: Box-Cox transform + lambda selection."""
from __future__ import annotations
import numpy as np, pandas as pd
import macroforecast as mf

def test_box_cox_lambda_loglik_lognormal_near_zero():
    rng = np.random.default_rng(0)
    x = pd.Series(np.exp(rng.normal(loc=1.0, scale=0.5, size=400)))  # log-normal -> lambda ~ 0
    lam = mf.preprocessing.box_cox_lambda(x, method="loglik")
    assert abs(lam) < 0.25
    # near-linear positive data -> lambda ~ 1
    lin = pd.Series(np.arange(1, 401, dtype=float) + rng.normal(scale=2, size=400))
    assert abs(mf.preprocessing.box_cox_lambda(lin, method="loglik") - 1.0) < 0.4

def test_box_cox_clean_roundtrip_and_records_lambda():
    rng = np.random.default_rng(1)
    panel = pd.DataFrame({
        "a": np.exp(rng.normal(size=200)),
        "b": np.arange(1, 201, dtype=float),
    })
    out = mf.preprocessing.box_cox_clean(panel)
    lambdas = out.attrs["macroforecast_box_cox_lambda"]
    assert set(lambdas) == {"a", "b"}
    for col in ("a", "b"):
        recon = mf.preprocessing.inverse_box_cox(out[col].to_numpy(), lambdas[col])
        np.testing.assert_allclose(recon, panel[col].to_numpy(), rtol=1e-6, atol=1e-6)

def test_box_cox_rejects_nonpositive_and_guerrero_runs():
    try:
        mf.preprocessing.box_cox_lambda(pd.Series([1.0, -2.0, 3.0, 4.0])); assert False
    except ValueError: pass
    g = mf.preprocessing.box_cox_lambda(pd.Series(np.arange(1, 121, dtype=float)),
                                        method="guerrero", period=12)
    assert -1.0 <= g <= 2.0
