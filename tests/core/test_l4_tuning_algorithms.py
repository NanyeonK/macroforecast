"""Issue #217 -- L4 search_algorithm dispatch."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macroforecast.core.runtime import _resolve_l4_tuning


def _toy_data(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    y = pd.Series(2.0 * X["a"] - 0.5 * X["b"] + rng.normal(scale=0.5, size=n))
    return X, y


def test_cv_path_picks_alpha_for_ridge():
    X, y = _toy_data()
    params = {"family": "ridge", "search_algorithm": "cv_path", "alpha": 1.0, "_l4_leaf_config": {}}
    result = _resolve_l4_tuning(params, X, y)
    assert "alpha" in result
    # Default alphas list: 6 candidates; the chosen one should be one of them
    # (or override via leaf_config).
    assert isinstance(result["alpha"], float)


def test_cv_path_uses_leaf_config_alphas():
    X, y = _toy_data()
    custom_alphas = [0.05, 0.5, 5.0]
    params = {
        "family": "lasso",
        "search_algorithm": "cv_path",
        "alpha": 1.0,
        "_l4_leaf_config": {"cv_path_alphas": custom_alphas},
    }
    result = _resolve_l4_tuning(params, X, y)
    assert result["alpha"] in custom_alphas


def test_grid_search_dispatch():
    X, y = _toy_data()
    params = {
        "family": "ridge",
        "search_algorithm": "grid_search",
        "alpha": 1.0,
        "_l4_leaf_config": {"tuning_grid": {"alpha": [0.1, 1.0, 10.0]}},
    }
    result = _resolve_l4_tuning(params, X, y)
    assert result["alpha"] in [0.1, 1.0, 10.0]


def test_random_search_dispatch_with_distributions():
    X, y = _toy_data()
    params = {
        "family": "ridge",
        "search_algorithm": "random_search",
        "alpha": 1.0,
        "random_state": 7,
        "_l4_leaf_config": {
            "tuning_distributions": {"alpha": [0.1, 0.5, 1.0, 2.0, 5.0]},
            "tuning_budget": 4,
        },
    }
    result = _resolve_l4_tuning(params, X, y)
    assert "alpha" in result


def test_bayesian_optimization_falls_back_to_random_when_optuna_missing():
    pytest.importorskip  # noqa: B018
    import importlib.util

    if importlib.util.find_spec("optuna") is not None:
        pytest.skip("optuna installed; this test pins the no-optuna fallback")
    X, y = _toy_data()
    params = {
        "family": "ridge",
        "search_algorithm": "bayesian_optimization",
        "alpha": 1.0,
        "random_state": 0,
        "_l4_leaf_config": {
            "tuning_distributions": {"alpha": [0.1, 1.0, 10.0]},
            "tuning_budget": 4,
        },
    }
    result = _resolve_l4_tuning(params, X, y)
    # Either the search succeeded or the data was too short -- either way
    # the call must return a params dict and not raise.
    assert isinstance(result, dict)
    assert "alpha" in result


def test_short_panel_returns_params_unchanged():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    y = pd.Series([0.1, 0.2, 0.3])
    params = {"family": "ridge", "search_algorithm": "grid_search", "alpha": 1.0}
    result = _resolve_l4_tuning(params, X, y)
    assert result["alpha"] == 1.0  # unchanged
