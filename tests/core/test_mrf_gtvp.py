"""macroeconomic_random_forest family backed by Ryan Lucas's MRF
reference implementation, vendored under
``macroforecast/_vendor/macro_random_forest/`` with surgical numpy 2.x /
pandas 2.x compatibility patches. Reference: Goulet Coulombe (2024)
"The Macroeconomy as a Random Forest" (arXiv:2006.12724); upstream:
https://github.com/RyanLucas3/MacroRandomForest.

Re-anchored from the in-house ``_MRFWrapper`` to ``_MRFExternalWrapper``
in v0.8.9 (see CHANGELOG honesty-pass entry and
``docs/architecture/v089_verification_results.md`` § V2.2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from macroforecast.core.runtime import _MRFExternalWrapper
from macroforecast.core.ops.l4_ops import (
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
        },
        index=pd.RangeIndex(n),
    )
    y = pd.Series(
        0.5 * X["x1"] + 0.3 * X["x2"] - 0.2 * X["x3"] + rng.normal(scale=0.1, size=n),
        index=X.index,
    )
    return X, y


def _fit_and_predict(model, X, y, n_train=60):
    train_X, test_X = X.iloc[:n_train], X.iloc[n_train:]
    train_y = y.iloc[:n_train]
    model.fit(train_X, train_y)
    return model.predict(test_X), train_X, test_X


def test_mrf_cached_betas_populated_after_predict():
    """``_MRFExternalWrapper`` caches the GTVP β series after predict so
    L7 ``mrf_gtvp`` can read them. Shape = (n_train + n_test, K + 1)."""

    X, y = _toy_panel()
    model = _MRFExternalWrapper(B=8, parallelise=False, n_cores=1, random_state=0)
    preds, _, test_X = _fit_and_predict(model, X, y, n_train=60)
    assert preds.shape == (len(test_X),)
    assert model._cached_betas is not None
    # 60 train + 20 test = 80 rows; intercept + 3 features = 4 cols.
    assert model._cached_betas.shape == (len(X), X.shape[1] + 1)
    # B bootstrap × n_oos forecast ensemble.
    assert model._cached_pred_ensemble is not None
    assert model._cached_pred_ensemble.shape == (8, len(test_X))


def test_mrf_predicts_finite_values_and_correct_shape():
    X, y = _toy_panel()
    model = _MRFExternalWrapper(B=10, parallelise=False, n_cores=1, random_state=0)
    preds, _, test_X = _fit_and_predict(model, X, y, n_train=60)
    assert preds.shape == (len(test_X),)
    assert np.all(np.isfinite(preds))


def test_mrf_seed_makes_predictions_deterministic():
    """``random_state`` propagates via ``np.random.seed`` before each
    construction (mrf-web does not expose a seed kwarg). Two wrappers
    with matching seeds produce matching forecasts."""

    X, y = _toy_panel()
    a = _MRFExternalWrapper(B=10, parallelise=False, n_cores=1, random_state=0)
    b = _MRFExternalWrapper(B=10, parallelise=False, n_cores=1, random_state=0)
    preds_a, _, _ = _fit_and_predict(a, X, y, n_train=60)
    preds_b, _, _ = _fit_and_predict(b, X, y, n_train=60)
    np.testing.assert_allclose(preds_a, preds_b, rtol=1e-12, atol=1e-12)


def test_mrf_distinct_seeds_produce_distinct_forecasts():
    X, y = _toy_panel()
    a = _MRFExternalWrapper(B=10, parallelise=False, n_cores=1, random_state=0)
    b = _MRFExternalWrapper(B=10, parallelise=False, n_cores=1, random_state=42)
    preds_a, _, _ = _fit_and_predict(a, X, y, n_train=60)
    preds_b, _, _ = _fit_and_predict(b, X, y, n_train=60)
    assert not np.allclose(preds_a, preds_b)


def test_mrf_handles_missing_columns_at_predict_time():
    """Test panel may drop a feature seen at fit; the wrapper aligns to
    the training column order and fills missing columns with 0."""

    X, y = _toy_panel()
    model = _MRFExternalWrapper(B=8, parallelise=False, n_cores=1, random_state=0)
    train_X = X.iloc[:60]
    train_y = y.iloc[:60]
    model.fit(train_X, train_y)
    test_X_partial = X.iloc[60:].drop(columns=["x3"])
    preds = model.predict(test_X_partial)
    assert preds.shape == (len(test_X_partial),)
    assert np.all(np.isfinite(preds))
