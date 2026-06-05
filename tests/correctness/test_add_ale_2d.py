"""ADD: accumulated_local_effect_2d (second-order ALE)."""
from __future__ import annotations
import numpy as np, pandas as pd
import macroforecast as mf

class _Model:
    def __init__(self, fn): self.fn = fn
    def predict(self, X): return np.asarray(self.fn(X), dtype=float)

def _X(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"a": rng.uniform(-2, 2, n), "b": rng.uniform(-2, 2, n),
                         "c": rng.normal(size=n)})

def test_additive_model_has_zero_interaction():
    X = _X()
    add = _Model(lambda d: 2.0 * d["a"].to_numpy() + 3.0 * d["b"].to_numpy())
    tab = mf.interpretation.accumulated_local_effect_2d(add, X, features=("a", "b"), bins=6)
    assert np.abs(tab["ale"].to_numpy()).max() < 1e-6        # no interaction -> ~0

def test_multiplicative_model_captures_interaction():
    X = _X(seed=1)
    mul = _Model(lambda d: d["a"].to_numpy() * d["b"].to_numpy())
    tab = mf.interpretation.accumulated_local_effect_2d(mul, X, features=("a", "b"), bins=8)
    # interaction surface tracks center_1 * center_2 (the true x1*x2 interaction)
    corr = np.corrcoef(tab["ale"], tab["center_1"] * tab["center_2"])[0, 1]
    assert corr > 0.9
    assert set(tab.columns) == {"bin_1", "bin_2", "center_1", "center_2", "ale", "count"}
