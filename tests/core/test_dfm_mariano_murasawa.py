"""Issue #188 -- Mariano-Murasawa-style DFM via statsmodels' Kalman state-space MLE."""
from __future__ import annotations

import numpy as np
import pandas as pd

from macrocast.core.runtime import _DFMMixedFrequency
from macrocast.core.ops.l4_ops import (
    OPERATIONAL_MODEL_FAMILIES,
    FUTURE_MODEL_FAMILIES,
    get_family_status,
)


def test_dfm_mariano_murasawa_is_operational():
    assert "dfm_mixed_mariano_murasawa" in OPERATIONAL_MODEL_FAMILIES
    assert "dfm_mixed_mariano_murasawa" not in FUTURE_MODEL_FAMILIES
    assert get_family_status("dfm_mixed_mariano_murasawa") == "operational"


def _toy_panel(n: int = 80, k: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(size=n)) * 0.3
    loadings = rng.normal(size=k)
    X = pd.DataFrame(
        np.outer(f, loadings) + rng.normal(scale=0.1, size=(n, k)),
        columns=[f"x{i+1}" for i in range(k)],
    )
    y = pd.Series(0.7 * f + rng.normal(scale=0.1, size=n), name="y")
    return X, y


def test_dfm_predicts_finite_values():
    X, y = _toy_panel()
    model = _DFMMixedFrequency(n_factors=1, factor_order=1).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert np.all(np.isfinite(preds))


def test_dfm_results_object_is_kalman_state_space():
    """The runtime must produce a real ``statsmodels`` Kalman result -- this
    is the central honesty pin. v0.1 returned a PCA + AR(1) shortcut.
    """

    X, y = _toy_panel()
    model = _DFMMixedFrequency(n_factors=1, factor_order=1).fit(X, y)
    # Either Kalman MLE landed and stored a results object, or the data was
    # too short and we fell back. The path we want to prove is the former.
    assert model._results is not None, "DFM should fit a real Kalman model on the toy panel"
    # statsmodels exposes ``filtered_state`` on a successful fit.
    assert hasattr(model._results, "filtered_state")


def test_dfm_handles_short_series_via_fallback():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(5, 2)), columns=["a", "b"])
    y = pd.Series(rng.normal(size=5))
    model = _DFMMixedFrequency(n_factors=1).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert np.all(np.isfinite(preds))


def test_dfm_factor_order_param_is_honoured():
    X, y = _toy_panel(n=120)
    a = _DFMMixedFrequency(n_factors=1, factor_order=1).fit(X, y)
    b = _DFMMixedFrequency(n_factors=1, factor_order=2).fit(X, y)
    # If both fit successfully, predictions should differ (different VAR
    # order on the latent factor).
    if a._results is not None and b._results is not None:
        # broadcast-uniform values, but should differ across configs.
        assert not np.allclose(a.predict(X)[0], b.predict(X)[0])
