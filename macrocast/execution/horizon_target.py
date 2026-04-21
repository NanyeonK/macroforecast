"""Target construction for 1.2.4 horizon_target_construction axis.

Provides forward-transform (build training target from raw y at horizon h) and
inverse-transform (convert model's forecast back to raw y-scale so metrics can
be computed on the original series).

Four constructions are operational in v1.0:

- ``future_level_y_t_plus_h``  : y_{t+h}                   (default, identity inverse)
- ``future_diff``              : y_{t+h} - y_t              inverse: y_t + ŷ
- ``future_logdiff``           : log(y_{t+h}) - log(y_t)    inverse: y_t * exp(ŷ)

All constructions share a single vectorised forward implementation.  The inverse
takes scalar ŷ (a point forecast) plus the anchor level y_t (value at the
origin index) and returns the predicted level y_{t+h}.
"""
from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd


OPERATIONAL_CONSTRUCTIONS: Final[frozenset[str]] = frozenset({
    "future_level_y_t_plus_h",
    "future_diff",
    "future_logdiff",
})


def _log_or_raise(series: pd.Series, *, construction: str) -> pd.Series:
    """log(series) with strict-positivity check; non-positive values make
    logdiff/cumulative_growth targets undefined."""
    if (series <= 0).any():
        raise ValueError(
            f"horizon_target_construction={construction!r} requires strictly "
            f"positive y values (got min={float(series.min())})"
        )
    return np.log(series)


def build_horizon_target(y: pd.Series, horizon: int, construction: str) -> pd.Series:
    """Forward transform: build the training target at horizon ``horizon`` from
    the raw series ``y``.  Output is aligned to ``y``'s index with NaN at the
    trailing ``horizon`` positions where y_{t+h} is not observed.
    """
    if construction not in OPERATIONAL_CONSTRUCTIONS:
        raise ValueError(
            f"unknown horizon_target_construction={construction!r}; "
            f"operational set is {sorted(OPERATIONAL_CONSTRUCTIONS)}"
        )
    y_future = y.shift(-horizon)
    if construction == "future_level_y_t_plus_h":
        return y_future
    if construction == "future_diff":
        return y_future - y
    log_y = _log_or_raise(y, construction=construction)
    log_y_future = _log_or_raise(y_future.dropna(), construction=construction).reindex(y.index)
    return log_y_future - log_y


def inverse_horizon_target(
    y_hat: float,
    y_anchor: float,
    construction: str,
) -> float:
    """Inverse transform: convert the model's forecast (on construction scale)
    back to the raw y-level y_{t+h}.  ``y_anchor`` is y at the origin index
    (the last observed level at forecast time)."""
    if construction not in OPERATIONAL_CONSTRUCTIONS:
        raise ValueError(
            f"unknown horizon_target_construction={construction!r}; "
            f"operational set is {sorted(OPERATIONAL_CONSTRUCTIONS)}"
        )
    y_hat_f = float(y_hat)
    if construction == "future_level_y_t_plus_h":
        return y_hat_f
    if construction == "future_diff":
        return float(y_anchor) + y_hat_f
    # logdiff
    if y_anchor <= 0:
        raise ValueError(
            f"horizon_target_construction={construction!r} inverse requires "
            f"strictly positive y_anchor (got {y_anchor!r})"
        )
    return float(y_anchor) * float(np.exp(y_hat_f))


def is_log_space(construction: str) -> bool:
    """True if the forecast scale is logarithmic (logdiff / cumulative_growth_to_h)."""
    return construction == "future_logdiff"


def forward_scalar(y_val: float, y_anchor: float, construction: str) -> float:
    """Apply forward transform to a single scalar value.

    Used at the central row-computation site to express predicted / actual
    levels on the construction scale so metrics land on that scale.
    """
    if construction not in OPERATIONAL_CONSTRUCTIONS:
        raise ValueError(
            f"unknown horizon_target_construction={construction!r}; "
            f"operational set is {sorted(OPERATIONAL_CONSTRUCTIONS)}"
        )
    y_val_f = float(y_val)
    if construction == "future_level_y_t_plus_h":
        return y_val_f
    if construction == "future_diff":
        return y_val_f - float(y_anchor)
    # logdiff
    if y_val_f <= 0 or float(y_anchor) <= 0:
        raise ValueError(
            f"horizon_target_construction={construction!r} forward requires "
            f"strictly positive y and y_anchor (got y={y_val_f!r}, "
            f"y_anchor={float(y_anchor)!r})"
        )
    return float(np.log(y_val_f) - np.log(float(y_anchor)))
