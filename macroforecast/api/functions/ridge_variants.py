"""Standalone ridge-variant family fit callables.

Exposes four specialized ridge regression fit functions beyond the standard
``ridge_fit`` (in ridge.py):

- ``nonneg_ridge_fit``             -- Non-negative ridge (Assemblage Regression).
- ``random_walk_ridge_fit``        -- Two-Stage Random Walk Ridge (Coulombe 2025 IJF).
- ``shrink_to_target_ridge_fit``   -- Shrink-to-Target Ridge (Albacore Variant A).
- ``fused_difference_ridge_fit``   -- Fused-Difference Ridge (Albacore Variant B).

Each callable builds the corresponding private runtime class directly and
returns a frozen ``RidgeFitResult`` (same class as ``ridge_fit``, imported from
``macroforecast.functions.ridge``) exposing ``coef_``, ``intercept_``,
``alpha``, and a ``.predict(X)`` method. Bit-exact with recipe-based ridge
when using the same parameter values.

Promoted in v0.9.5: L4 ridge-variant standalone callables (4 ops).
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from .ridge import RidgeFitResult


# ---------------------------------------------------------------------------
# Shared input normalisation
# ---------------------------------------------------------------------------

def _prep_xy(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Convert X/y to DataFrame/Series with default column names."""
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    if isinstance(y, np.ndarray):
        y = pd.Series(y.ravel(), name="y")
    return X, y


def _extract_coef(model: Any, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Extract coef_ / intercept_ from any fitted ridge-variant model."""
    # Try attributes in priority order; each may be None or a numpy array.
    raw: Any = None
    for attr in ("_coef", "coef_"):
        val = getattr(model, attr, None)
        if val is not None:
            raw = val
            break
    if raw is None:
        raw = np.zeros(X.shape[1])
    coef = np.asarray(raw, dtype=float)
    intercept = float(getattr(model, "_intercept", getattr(model, "intercept_", 0.0)))
    return coef, intercept


# ---------------------------------------------------------------------------
# nonneg_ridge_fit
# ---------------------------------------------------------------------------

def nonneg_ridge_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    alpha: float = 1.0,
) -> RidgeFitResult:
    """Standalone non-negative ridge regression (Coulombe et al. 2024).

    Solves ``min ||y - Xβ||² + α||β||²`` subject to β >= 0 via
    ``scipy.optimize.nnls`` on the augmented system. Bit-exact with
    recipe-based ``ridge(coefficient_constraint="nonneg")``.

    Parameters
    ----------
    X :
        Feature matrix. Shape (n_samples, n_features). Accepts numpy
        arrays or DataFrames.
    y :
        Target vector. Shape (n_samples,).
    alpha : float
        L2 regularisation strength. Must be >= 0. Default 1.0.

    Returns
    -------
    RidgeFitResult
        Fitted result exposing ``coef_`` (non-negative), ``intercept_``,
        ``alpha``, and a ``.predict(X)`` method.

    References
    ----------
    Coulombe, Klieber, Barrette, Goebel (2024) "Maximally Forward-Looking
    Core Inflation." Journal of Applied Econometrics.
    """
    from macroforecast.core.runtime import _NonNegRidge

    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha!r}")

    X, y = _prep_xy(X, y)
    model = _NonNegRidge(alpha=float(alpha))
    model.fit(X, y)
    coef, intercept = _extract_coef(model, X)
    return RidgeFitResult(coef_=coef, intercept_=intercept, alpha=float(alpha), _model=model)


# ---------------------------------------------------------------------------
# random_walk_ridge_fit
# ---------------------------------------------------------------------------

def random_walk_ridge_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    alpha: float = 1.0,
    vol_model: Literal["garch11", "ewma"] = "garch11",
    max_alpha_ratio: float = 1e6,
    alpha_search_policy: Literal["second_cv", "fixed"] = "second_cv",
    alpha_grid: list[float] | None = None,
    cv_folds: int = 5,
    random_state: int = 0,
) -> RidgeFitResult:
    """Standalone Two-Stage Random Walk Ridge (Coulombe 2025 IJF).

    Implements "Time-Varying Parameters as Ridge Regressions" (Coulombe 2025
    IJF). Two closed-form steps: step 1 uses the homogeneous-variance ridge
    (Eq. 9); step 2 refits with heterogeneous variances estimated from
    step-1 residuals via a volatility model (Eq. 11). Bit-exact with
    recipe-based ``ridge(prior="random_walk")``.

    Parameters
    ----------
    X :
        Feature matrix. Shape (n_samples, n_features).
    y :
        Target vector. Shape (n_samples,).
    alpha : float
        Initial L2 regularisation strength for step 1. Default 1.0.
    vol_model : str
        Volatility model for the Omega reconstruction: "garch11" (default;
        paper spec) or "ewma" (RiskMetrics lambda=0.94; no extra deps).
    max_alpha_ratio : float
        Upper bound on step-2 alpha / step-1 alpha ratio. Default 1e6.
    alpha_search_policy : str
        "second_cv" (default; re-runs CV after warm-start step 1) or
        "fixed" (uses ``alpha`` as-is for both steps).
    alpha_grid : list[float] or None
        Grid for the second-step CV. Defaults to [0.01, 0.1, 1, 10, 100].
    cv_folds : int
        Number of CV folds for the second-step lambda selection. Default 5.
    random_state : int
        RNG seed. Default 0.

    Returns
    -------
    RidgeFitResult
        Fitted result exposing ``coef_`` (most-recent TVP coefficient
        vector β_T), ``intercept_``, ``alpha``, and a ``.predict(X)`` method.

    References
    ----------
    Coulombe (2025) "Time-Varying Parameters as Ridge Regressions."
    International Journal of Forecasting.
    """
    from macroforecast.core.runtime import _TwoStageRandomWalkRidge

    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha!r}")

    X, y = _prep_xy(X, y)
    model = _TwoStageRandomWalkRidge(
        alpha=float(alpha),
        vol_model=vol_model,
        max_alpha_ratio=float(max_alpha_ratio),
        alpha_search_policy=alpha_search_policy,
        alpha_grid=alpha_grid,
        cv_folds=cv_folds,
        random_state=random_state,
    )
    model.fit(X, y)
    # For TVP ridge, extract the most-recent coefficient vector (_beta_last).
    _beta_last = getattr(model, "_beta_last", None)
    coef = np.asarray(_beta_last if _beta_last is not None else np.zeros(X.shape[1]), dtype=float)
    intercept = float(getattr(model, "_intercept", 0.0))
    return RidgeFitResult(coef_=coef, intercept_=intercept, alpha=float(alpha), _model=model)


# ---------------------------------------------------------------------------
# shrink_to_target_ridge_fit
# ---------------------------------------------------------------------------

def shrink_to_target_ridge_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    alpha: float = 1.0,
    prior_target: Any = None,
    simplex: bool = True,
    nonneg: bool = True,
) -> RidgeFitResult:
    """Standalone Shrink-to-Target Ridge (Albacore Variant A).

    Solves ``min ||y - Xw||² + α||w - w_target||²`` subject to
    w >= 0 (when ``nonneg=True``) and w'1 = 1 (when ``simplex=True``).
    Bit-exact with recipe-based ``ridge(prior="shrink_to_target")``.

    Parameters
    ----------
    X :
        Feature matrix. Shape (n_samples, n_features).
    y :
        Target vector. Shape (n_samples,).
    alpha : float
        Shrinkage strength. Default 1.0.
    prior_target : array-like or None
        Target coefficient vector. If None, defaults to uniform ``1/K``
        internally. Must be supplied for paper-faithful Albacore behavior
        (raises ValueError if None when ``simplex=True``).
    simplex : bool
        Enforce simplex constraint (w >= 0 and sum w = 1). Default True.
    nonneg : bool
        Enforce non-negativity without simplex normalization. Default True.

    Returns
    -------
    RidgeFitResult
        Fitted result exposing ``coef_``, ``intercept_``, ``alpha``, and
        a ``.predict(X)`` method.

    References
    ----------
    Coulombe, Klieber, Barrette, Goebel (2024) "Maximally Forward-Looking
    Core Inflation." Albacore_comps (Variant A).
    """
    from macroforecast.core.runtime import _ShrinkToTargetRidge

    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha!r}")

    X, y = _prep_xy(X, y)
    model = _ShrinkToTargetRidge(
        alpha=float(alpha),
        prior_target=prior_target,
        simplex=simplex,
        nonneg=nonneg,
    )
    model.fit(X, y)
    coef, intercept = _extract_coef(model, X)
    return RidgeFitResult(coef_=coef, intercept_=intercept, alpha=float(alpha), _model=model)


# ---------------------------------------------------------------------------
# fused_difference_ridge_fit
# ---------------------------------------------------------------------------

def fused_difference_ridge_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    alpha: float = 1.0,
    difference_order: int = 1,
    mean_equality: bool = True,
    nonneg: bool = True,
) -> RidgeFitResult:
    """Standalone Fused-Difference Ridge (Albacore Variant B).

    Solves ``min ||y - Xw||² + α||Dw||²`` where D is the difference
    operator of order ``difference_order``. Optionally enforces w >= 0
    (``nonneg=True``) and mean-equality (``mean_equality=True``). Bit-exact
    with recipe-based ``ridge(prior="fused_difference")``.

    Parameters
    ----------
    X :
        Feature matrix. Shape (n_samples, n_features).
    y :
        Target vector. Shape (n_samples,).
    alpha : float
        Fusion penalty strength. Default 1.0.
    difference_order : int
        Order of the difference operator D. Default 1 (first differences).
    mean_equality : bool
        Enforce mean-equality constraint across coefficient groups. Default True.
    nonneg : bool
        Enforce non-negativity constraint. Default True.

    Returns
    -------
    RidgeFitResult
        Fitted result exposing ``coef_``, ``intercept_``, ``alpha``, and
        a ``.predict(X)`` method.

    References
    ----------
    Coulombe, Klieber, Barrette, Goebel (2024) "Maximally Forward-Looking
    Core Inflation." Albacore_ranks (Variant B).
    """
    from macroforecast.core.runtime import _FusedDifferenceRidge

    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha!r}")

    X, y = _prep_xy(X, y)
    model = _FusedDifferenceRidge(
        alpha=float(alpha),
        difference_order=difference_order,
        mean_equality=mean_equality,
        nonneg=nonneg,
    )
    model.fit(X, y)
    coef, intercept = _extract_coef(model, X)
    return RidgeFitResult(coef_=coef, intercept_=intercept, alpha=float(alpha), _model=model)


__all__ = [
    "nonneg_ridge_fit",
    "random_walk_ridge_fit",
    "shrink_to_target_ridge_fit",
    "fused_difference_ridge_fit",
]
