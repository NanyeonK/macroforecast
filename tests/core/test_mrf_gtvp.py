"""Issue #187 -- macroeconomic_random_forest implements Coulombe (2024)
GTVP via per-leaf local linear regressions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from macrocast.core.runtime import _MRFWrapper
from macrocast.core.ops.l4_ops import (
    OPERATIONAL_MODEL_FAMILIES,
    FUTURE_MODEL_FAMILIES,
    get_family_status,
)


def test_macroeconomic_random_forest_is_operational():
    assert "macroeconomic_random_forest" in OPERATIONAL_MODEL_FAMILIES
    assert "macroeconomic_random_forest" not in FUTURE_MODEL_FAMILIES
    assert get_family_status("macroeconomic_random_forest") == "operational"


def _toy_panel(n: int = 80, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.normal(size=n),
        }
    )
    y = pd.Series(0.5 * X["x1"] + 0.3 * X["x2"] - 0.2 * X["x3"] + rng.normal(scale=0.1, size=n))
    return X, y


def test_mrf_per_leaf_models_populated():
    X, y = _toy_panel()
    model = _MRFWrapper(n_estimators=10, max_depth=3, random_state=0).fit(X, y)
    # Every tree should have at least one leaf with a fitted local LinearRegression.
    assert len(model._leaf_models) == 10
    has_linear_leaf = any(
        any(isinstance(v, LinearRegression) for v in tree.values())
        for tree in model._leaf_models
    )
    assert has_linear_leaf, "expected at least one leaf with a local LinearRegression"


def test_mrf_predicts_finite_values_and_correct_shape():
    X, y = _toy_panel()
    model = _MRFWrapper(n_estimators=15, max_depth=4, random_state=0).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert np.all(np.isfinite(preds))


def test_mrf_seed_makes_predictions_deterministic():
    X, y = _toy_panel()
    a = _MRFWrapper(n_estimators=10, max_depth=3, random_state=0).fit(X, y)
    b = _MRFWrapper(n_estimators=10, max_depth=3, random_state=0).fit(X, y)
    np.testing.assert_allclose(a.predict(X), b.predict(X))


def test_mrf_distinct_seeds_produce_distinct_forests():
    X, y = _toy_panel()
    a = _MRFWrapper(n_estimators=10, max_depth=3, random_state=0).fit(X, y)
    b = _MRFWrapper(n_estimators=10, max_depth=3, random_state=42).fit(X, y)
    assert not np.allclose(a.predict(X), b.predict(X))


def test_mrf_handles_missing_columns_at_predict_time():
    X, y = _toy_panel()
    model = _MRFWrapper(n_estimators=10, max_depth=3, random_state=0).fit(X, y)
    X_partial = X.drop(columns=["x3"])
    preds = model.predict(X_partial)
    assert preds.shape == (len(X_partial),)
    assert np.all(np.isfinite(preds))
