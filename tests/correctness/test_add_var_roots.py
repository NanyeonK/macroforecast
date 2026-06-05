"""ADD: var_roots (VAR companion-matrix stability)."""
from __future__ import annotations
import numpy as np, pandas as pd
import macroforecast as mf

def _sim(A, n=400, seed=0):
    rng = np.random.default_rng(seed); k = A.shape[0]; y = np.zeros((n, k))
    for t in range(1, n):
        y[t] = A @ y[t-1] + rng.normal(scale=0.5, size=k)
    return pd.DataFrame(y, columns=[f"v{i}" for i in range(k)],
                        index=pd.date_range("1990-01-31", periods=n, freq="ME"))

def test_var_roots_stable_and_eigenvalues():
    A = np.array([[0.5, 0.1], [0.0, 0.4]])
    fit = mf.models.var(_sim(A), target="v0", n_lag=1)
    out = mf.models.var_roots(fit)
    assert out["is_stable"] is True
    assert out["max_modulus"] < 1.0
    # VAR(1): companion == A, so the moduli are |eig(A)| ~ {0.5, 0.4}
    assert abs(out["max_modulus"] - 0.5) < 0.15
    assert out["n_lag"] == 1 and out["n_vars"] == 2
