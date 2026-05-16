"""Standalone ridge regression wrapper.

Exposes ``ridge_fit`` -- a sklearn-style callable that calls the L4 ridge
dispatch (``_build_l4_model``) directly, bypassing recipe machinery.

Bit-exact compatible with recipe-based ridge: the same estimator classes
(``sklearn.linear_model.Ridge``, ``_NonNegRidge``, ``_TwoStageRandomWalkRidge``,
``_ShrinkToTargetRidge``, ``_FusedDifferenceRidge``) are used under the hood.

Cycle 22 POC -- pattern validates for L4/L3/L5/L6/L7 expansion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RidgeFitResult:
    """Result of ``ridge_fit``.

    Attributes:
        coef_:       Fitted coefficient vector (1-D, length = n_features).
        intercept_:  Fitted intercept scalar.
        alpha:       Regularisation strength used.
        _model:      The underlying fitted estimator (internal; for ``.predict``).
    """

    coef_: np.ndarray
    intercept_: float
    alpha: float
    _model: Any  # internal; not part of public contract

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict using the fitted model.

        Parameters
        ----------
        X:
            Feature matrix.  Accepts numpy arrays or DataFrames.  When
            ``X`` is a bare array, columns are assumed to match the training
            order.

        Returns
        -------
        np.ndarray
            1-D float array of predictions.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        return np.asarray(self._model.predict(X), dtype=float)


def ridge_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    alpha: float = 1.0,
    prior: Literal["none", "random_walk", "shrink_to_target", "fused_difference"] = "none",
    coefficient_constraint: Literal["none", "nonneg"] = "none",
    vol_model: Literal["ewma", "garch11"] | None = None,
    random_state: int | None = None,
) -> RidgeFitResult:
    """Standalone ridge regression.

    Calls the L4 ridge family adapter directly; bypasses the recipe DAG.
    Produces bit-exact numeric output as recipe-based ridge with the same
    parameter values.

    Parameters
    ----------
    X:
        Feature matrix.  Shape (n_samples, n_features).  Accepts numpy
        arrays or DataFrames.
    y:
        Target vector.  Shape (n_samples,).  Accepts numpy arrays or Series.
    alpha:
        L2 regularisation strength.  Must be >= 0.  Default 1.0.
    prior:
        Coefficient prior.  ``"none"`` (default) = standard closed-form
        ridge.  ``"random_walk"`` = Goulet Coulombe (2025 IJF) TVP
        two-step estimator.  ``"shrink_to_target"`` = Albacore_comps
        Variant A.  ``"fused_difference"`` = Albacore_ranks Variant B.
    coefficient_constraint:
        Sign / cone constraint applied on top of the ridge penalty.
        ``"nonneg"`` enforces beta >= 0 via augmented NNLS.  Ignored
        when ``prior`` is ``"shrink_to_target"`` or
        ``"fused_difference"`` (those priors handle non-negativity
        internally).
    vol_model:
        Volatility model for the random-walk estimator's step-2 Omega
        reconstruction.  ``"ewma"`` (RiskMetrics lambda=0.94, no extra
        deps) or ``"garch11"`` (requires ``arch>=5.0``; auto-falls back
        to EWMA if unavailable).  Only used when ``prior="random_walk"``.
    random_state:
        Integer seed for stochastic sub-steps (reserved; currently unused
        in the standard ridge path).

    Returns
    -------
    RidgeFitResult
        Fitted result exposing ``coef_``, ``intercept_``, ``alpha``, and
        a ``.predict(X)`` method.
    """
    from ..core.runtime import _build_l4_model  # lazy import to avoid circular

    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha!r}")

    # Convert to DataFrame / Series so internal adapters have column info.
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    if isinstance(y, np.ndarray):
        y = pd.Series(y.ravel(), name="y")

    # Build the same estimator that the recipe DAG would build.
    params: dict[str, Any] = {
        "alpha": float(alpha),
        "prior": prior,
        "coefficient_constraint": coefficient_constraint,
    }
    if vol_model is not None:
        params["vol_model"] = vol_model
    if random_state is not None:
        params["random_state"] = int(random_state)

    model = _build_l4_model("ridge", params)
    model.fit(X, y)

    # Extract coef_ / intercept_ from the underlying estimator.
    coef = np.asarray(getattr(model, "coef_", np.zeros(X.shape[1])), dtype=float)
    intercept = float(getattr(model, "intercept_", 0.0))

    return RidgeFitResult(
        coef_=coef,
        intercept_=intercept,
        alpha=float(alpha),
        _model=model,
    )
