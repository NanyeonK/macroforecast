"""ADD: var_select_order (VARselect lag-order selection)."""
from __future__ import annotations
import numpy as np, pandas as pd
import macroforecast as mf


def test_var_select_order_structure_and_small_order_on_var1():
    rng = np.random.default_rng(0)
    n = 300
    y = np.zeros((n, 2))
    A = np.array([[0.5, 0.1], [0.0, 0.4]])
    for t in range(1, n):
        y[t] = A @ y[t - 1] + rng.normal(size=2)
    panel = pd.DataFrame(y, columns=["a", "b"],
                         index=pd.date_range("1990-01-31", periods=n, freq="ME"))
    out = mf.models.var_select_order(panel, maxlags=8, trend="c")
    sel = out["selected_orders"]
    assert set(sel) == {"aic", "bic", "hqic", "fpe"}
    assert all(0 <= int(v) <= 8 for v in sel.values())
    # BIC is conservative: a true VAR(1) should not need a high order.
    assert sel["bic"] <= 2
    assert "ics" in out and out["n_vars"] == 2
