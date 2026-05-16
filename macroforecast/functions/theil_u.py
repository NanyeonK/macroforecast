"""Standalone Theil inequality coefficient functions.

Exposes ``theil_u1`` and ``theil_u2`` as pure-numeric functions that can be
called without constructing a recipe.  The formulas are extracted from
``_add_l5_extended_metrics`` in ``macroforecast.core.runtime`` and produce
bit-exact identical results to recipe-based L5 evaluation.

Cycle 22 POC -- pattern validates for L5 metric expansion.

References
----------
Theil (1966) *Applied Economic Forecasting*, North-Holland.
"""
from __future__ import annotations

import numpy as np


def theil_u1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Theil's U1 inequality coefficient.

    U1 = sqrt(mean((y_true - y_pred)^2)) /
         (sqrt(mean(y_true^2)) + sqrt(mean(y_pred^2)))

    Bounded in [0, 1]: 0 = perfect forecast; 1 = worst possible.

    Produces bit-exact the same value as recipe-based L5 ``theil_u1``
    (extracted from ``_add_l5_extended_metrics``).

    Parameters
    ----------
    y_true:
        Actual (realised) values.  1-D float array of length N.
    y_pred:
        Forecast values.  Must be the same length as ``y_true``.

    Returns
    -------
    float
        U1 statistic, or ``nan`` when the denominator is zero (both
        ``y_true`` and ``y_pred`` are identically zero).

    Raises
    ------
    ValueError
        When ``y_true`` and ``y_pred`` have different lengths or are empty.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-D arrays.")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length; "
            f"got {len(y_true)} vs {len(y_pred)}."
        )
    if len(y_true) == 0:
        raise ValueError("y_true and y_pred must be non-empty.")

    rmse_forecast = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = float(np.sqrt(np.mean(y_true ** 2)) + np.sqrt(np.mean(y_pred ** 2)))
    return rmse_forecast / denom if denom > 0 else float("nan")


def theil_u2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prev: np.ndarray,
) -> float:
    """Theil's U2 inequality coefficient (ratio to no-change benchmark).

    U2 = sqrt(sum((y_pred - y_true)^2 / y_prev^2)) /
         sqrt(sum(((y_true - y_prev) / y_prev)^2))

    U2 < 1 means the forecast beats the random-walk (no-change) benchmark.
    Returns ``nan`` when fewer than two observations have a valid ``y_prev``,
    or when the denominator is zero.

    Produces bit-exact the same value as recipe-based L5 ``theil_u2``
    (extracted from ``_add_l5_extended_metrics``).

    Parameters
    ----------
    y_true:
        Actual (realised) values at time t.  1-D float array.
    y_pred:
        Forecast values at time t.  Same length as ``y_true``.
    y_prev:
        Actual values at time t-1 (the random-walk baseline).  Same
        length as ``y_true``.  Pass ``nan`` for observations where the
        previous value is unavailable; those rows are excluded.

    Returns
    -------
    float
        U2 statistic, or ``nan`` when insufficient data or denominator = 0.

    Raises
    ------
    ValueError
        When arrays have inconsistent lengths or are empty.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_prev = np.asarray(y_prev, dtype=float)

    if not (y_true.ndim == y_pred.ndim == y_prev.ndim == 1):
        raise ValueError("y_true, y_pred, and y_prev must be 1-D arrays.")
    if not (len(y_true) == len(y_pred) == len(y_prev)):
        raise ValueError("y_true, y_pred, and y_prev must have the same length.")
    if len(y_true) == 0:
        raise ValueError("Arrays must be non-empty.")

    # Restrict to rows where y_prev is available (not NaN).
    valid = ~np.isnan(y_prev)
    if valid.sum() < 2:
        return float("nan")

    yt = y_true[valid]
    yp = y_pred[valid]
    yp_prev = y_prev[valid]

    # Avoid division by zero from zero y_prev.
    safe_prev = np.where(np.abs(yp_prev) > 0, yp_prev, float("nan"))
    num_u2 = float(np.nansum(((yp - yt) / safe_prev) ** 2))
    den_u2 = float(np.nansum(((yt - yp_prev) / safe_prev) ** 2))
    return float(np.sqrt(num_u2 / den_u2)) if den_u2 > 0 else float("nan")
