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


# ---------------------------------------------------------------------------
# Coulombe-Surprenant-Leroux-Stevanovic (2022 JAE) Feature 3 schemes:
# kfold / poos / aic / bic. v0.9.0a0 audit gap-fix verifies each is wired
# rather than silently discarded.
# ---------------------------------------------------------------------------


def test_kfold_search_picks_alpha_from_grid():
    """``search_algorithm='kfold'`` was previously a silent no-op (string
    not in SEARCH_ALGORITHMS enum). The audit-fix wires KFold(shuffle)
    cross-validation; the resolved ``alpha`` must come from the
    candidate grid."""

    X, y = _toy_data()
    custom_alphas = [0.01, 0.1, 1.0, 10.0]
    params = {
        "family": "ridge", "search_algorithm": "kfold", "alpha": 99.0,
        "random_state": 1, "_l4_leaf_config": {"cv_path_alphas": custom_alphas},
    }
    result = _resolve_l4_tuning(params, X, y)
    assert result["alpha"] in custom_alphas
    # And it actually changed away from the input default (was 99.0).
    assert result["alpha"] != 99.0


def test_poos_search_picks_alpha_via_holdout():
    """POOS = pseudo-out-of-sample expanding-window holdout (paper §3.3).
    Distinct branch from kfold; same alpha-grid surface."""

    X, y = _toy_data()
    custom_alphas = [0.01, 0.1, 1.0, 10.0]
    params = {
        "family": "lasso", "search_algorithm": "poos", "alpha": 99.0,
        "random_state": 1, "_l4_leaf_config": {"cv_path_alphas": custom_alphas},
    }
    result = _resolve_l4_tuning(params, X, y)
    assert result["alpha"] in custom_alphas
    assert result["alpha"] != 99.0


def test_aic_search_picks_alpha_by_information_criterion():
    """AIC = n·log(RSS/n) + 2·k. Closed-form per-alpha, no cross-validation."""

    X, y = _toy_data()
    custom_alphas = [0.01, 0.1, 1.0, 10.0]
    params = {
        "family": "ridge", "search_algorithm": "aic", "alpha": 99.0,
        "_l4_leaf_config": {"cv_path_alphas": custom_alphas},
    }
    result = _resolve_l4_tuning(params, X, y)
    assert result["alpha"] in custom_alphas
    assert result["alpha"] != 99.0


def test_bic_search_picks_alpha_by_information_criterion():
    """BIC differs from AIC only in the penalty term log(n) vs 2; on a
    moderate-n panel they typically pick different alphas."""

    X, y = _toy_data(n=80, seed=4)
    custom_alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    leaf = {"cv_path_alphas": custom_alphas}
    params_aic = {
        "family": "ridge", "search_algorithm": "aic", "alpha": 99.0,
        "_l4_leaf_config": leaf,
    }
    params_bic = {
        "family": "ridge", "search_algorithm": "bic", "alpha": 99.0,
        "_l4_leaf_config": leaf,
    }
    res_aic = _resolve_l4_tuning(params_aic, X, y)
    res_bic = _resolve_l4_tuning(params_bic, X, y)
    assert res_aic["alpha"] in custom_alphas
    assert res_bic["alpha"] in custom_alphas


def test_paper16_cv_schemes_all_in_search_algorithms_enum():
    """The four paper-16 CV schemes must be registered in
    ``SEARCH_ALGORITHMS`` so the validator does not hard-reject recipes
    that name them. Audit gap-fix: previously the strings were silently
    dropped at runtime and the validator's options enum did not list
    them either."""

    from macroforecast.core.ops.l4_ops import SEARCH_ALGORITHMS
    for scheme in ("kfold", "poos", "aic", "bic"):
        assert scheme in SEARCH_ALGORITHMS, f"{scheme!r} missing from SEARCH_ALGORITHMS"


def test_non_alpha_family_passes_through_kfold_unchanged():
    """For families without an alpha hyperparameter (random_forest,
    svr_rbf, kernel_ridge), the new CV branches should pass through
    cleanly rather than crash."""

    X, y = _toy_data()
    params = {
        "family": "random_forest", "search_algorithm": "kfold",
        "n_estimators": 50, "max_depth": 4, "random_state": 0,
        "_l4_leaf_config": {},
    }
    result = _resolve_l4_tuning(params, X, y)
    # Audit fix #15: now RF actually picks max_depth from a grid; verify
    # n_estimators passes through and max_depth is one of the grid values.
    assert result["n_estimators"] == 50
    assert result["max_depth"] in (None, 4, 8, 16)


def test_kfold_kernel_ridge_uses_alpha_grid():
    """KRR is alpha-tunable; CV scheme picks alpha just like ridge."""

    X, y = _toy_data()
    custom_alphas = [0.01, 0.1, 1.0]
    params = {
        "family": "kernel_ridge", "search_algorithm": "kfold", "alpha": 99.0,
        "kernel": "rbf", "random_state": 1,
        "_l4_leaf_config": {"cv_path_alphas": custom_alphas},
    }
    result = _resolve_l4_tuning(params, X, y)
    assert result["alpha"] in custom_alphas
    assert result["alpha"] != 99.0


def test_kfold_random_forest_picks_max_depth_from_grid():
    """RF CV: pick ``max_depth`` from grid (None / 4 / 8 / 16 by default)."""

    X, y = _toy_data()
    params = {
        "family": "random_forest", "search_algorithm": "kfold",
        "n_estimators": 50, "max_depth": 99, "random_state": 0,
        "_l4_leaf_config": {"rf_max_depth_grid": [2, 4, 8]},
    }
    result = _resolve_l4_tuning(params, X, y)
    assert result["max_depth"] in (2, 4, 8)


def test_poos_svr_picks_C_from_grid():
    """SVR CV: pick ``C`` from grid (default {0.1, 1.0, 10.0})."""

    X, y = _toy_data()
    params = {
        "family": "svr_rbf", "search_algorithm": "poos",
        "C": 99.0, "epsilon": 0.1, "random_state": 0,
        "_l4_leaf_config": {"svr_C_grid": [0.5, 5.0, 50.0]},
    }
    result = _resolve_l4_tuning(params, X, y)
    assert result["C"] in (0.5, 5.0, 50.0)
    assert result["C"] != 99.0
