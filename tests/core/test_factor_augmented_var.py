"""Issue #184 -- FAVAR (Bernanke-Boivin-Eliasz 2005) promoted to operational."""
from __future__ import annotations

import numpy as np
import pandas as pd

from macroforecast.core.runtime import _FactorAugmentedVAR
from macroforecast.core.ops.l4_ops import (
    OPERATIONAL_MODEL_FAMILIES,
    FUTURE_MODEL_FAMILIES,
    get_family_status,
)


def test_factor_augmented_var_is_operational():
    assert "factor_augmented_var" in OPERATIONAL_MODEL_FAMILIES
    assert "factor_augmented_var" not in FUTURE_MODEL_FAMILIES
    assert get_family_status("factor_augmented_var") == "operational"


def _toy_panel(n: int = 80, k: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    # Simulate 2 latent factors driving k observables + a target.
    f1 = np.cumsum(rng.normal(size=n))
    f2 = np.cumsum(rng.normal(size=n))
    loadings = rng.normal(size=(2, k))
    X = pd.DataFrame(
        np.column_stack([f1, f2]) @ loadings + rng.normal(scale=0.1, size=(n, k)),
        columns=[f"x{i+1}" for i in range(k)],
    )
    y = pd.Series(0.5 * f1 + 0.3 * f2 + rng.normal(scale=0.1, size=n), name="y")
    return X, y


def test_favar_predicts_finite_values():
    X, y = _toy_panel()
    model = _FactorAugmentedVAR(p=2, n_factors=2).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert np.all(np.isfinite(preds))


def test_favar_reduces_dimension_via_factors():
    X, y = _toy_panel()
    model = _FactorAugmentedVAR(p=2, n_factors=2).fit(X, y)
    # Loadings matrix should reflect k columns -> n_factors components.
    assert model._loadings is not None
    assert model._loadings.shape == (2, X.shape[1])


def test_favar_handles_short_series_gracefully():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(5, 4)), columns=["a", "b", "c", "d"])
    y = pd.Series(rng.normal(size=5))
    model = _FactorAugmentedVAR(p=2, n_factors=2).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    # Falls through to a plain VAR/linear fallback path; still finite.
    assert np.all(np.isfinite(preds))
