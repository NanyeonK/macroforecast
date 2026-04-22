"""Target construction for 1.2.4 horizon_target_construction axis.

Provides forward-transform (build training target from the raw target series at
horizon h) and inverse-transform (convert model forecasts back to the raw
target scale so metrics can be computed on the original series).

Six constructions are operational in v1.0:

- ``future_target_level_t_plus_h``: target_{t+h} (default, identity inverse)
- ``future_diff``: target_{t+h} - target_t
- ``future_logdiff``: log(target_{t+h}) - log(target_t)
- ``average_growth_1_to_h``: (target_{t+h} / target_t - 1) / h
- ``average_difference_1_to_h``: (target_{t+h} - target_t) / h
- ``average_log_growth_1_to_h``: (log(target_{t+h}) - log(target_t)) / h

All constructions share a single vectorised forward implementation. The inverse
takes a scalar point forecast plus the anchor target level at the origin index
and returns the predicted target level at t+h.
"""
from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd


OPERATIONAL_CONSTRUCTIONS: Final[frozenset[str]] = frozenset({
    "future_target_level_t_plus_h",
    "future_diff",
    "future_logdiff",
    "average_growth_1_to_h",
    "average_difference_1_to_h",
    "average_log_growth_1_to_h",
})
LEGACY_CONSTRUCTION_ALIASES: Final[dict[str, str]] = {
    "future_level_y_t_plus_h": "future_target_level_t_plus_h",
}
SUPPORTED_CONSTRUCTIONS: Final[frozenset[str]] = (
    OPERATIONAL_CONSTRUCTIONS | frozenset(LEGACY_CONSTRUCTION_ALIASES)
)


def canonicalize_horizon_target_construction(construction: str) -> str:
    """Return the canonical target-construction id for legacy aliases."""
    return LEGACY_CONSTRUCTION_ALIASES.get(str(construction), str(construction))


def _log_or_raise(series: pd.Series, *, construction: str) -> pd.Series:
    """log(series) with strict-positivity check."""
    if (series <= 0).any():
        raise ValueError(
            f"horizon_target_construction={construction!r} requires strictly "
            f"positive target values (got min={float(series.min())})"
        )
    return np.log(series)


def _positive_horizon(horizon: int) -> int:
    horizon_i = int(horizon)
    if horizon_i <= 0:
        raise ValueError(f"horizon must be positive, got {horizon!r}")
    return horizon_i


def _require_positive_scalar(value: float, *, construction: str, name: str) -> float:
    value_f = float(value)
    if value_f <= 0:
        raise ValueError(
            f"horizon_target_construction={construction!r} requires strictly "
            f"positive {name} (got {value_f!r})"
        )
    return value_f


def build_horizon_target(target: pd.Series, horizon: int, construction: str) -> pd.Series:
    """Build the training target at ``horizon`` from the raw target series.

    Output is aligned to the target index with NaN at the trailing ``horizon``
    positions where target_{t+h} is not observed.
    """
    horizon = _positive_horizon(horizon)
    construction = canonicalize_horizon_target_construction(construction)
    if construction not in OPERATIONAL_CONSTRUCTIONS:
        raise ValueError(
            f"unknown horizon_target_construction={construction!r}; "
            f"operational set is {sorted(SUPPORTED_CONSTRUCTIONS)}"
        )
    target_future = target.shift(-horizon)
    if construction == "future_target_level_t_plus_h":
        return target_future
    if construction == "future_diff":
        return target_future - target
    if construction == "average_difference_1_to_h":
        return (target_future - target) / horizon
    if construction == "average_growth_1_to_h":
        _log_or_raise(target, construction=construction)
        _log_or_raise(target_future.dropna(), construction=construction)
        return (target_future / target - 1.0) / horizon
    log_target = _log_or_raise(target, construction=construction)
    if construction == "future_logdiff":
        log_target_future = _log_or_raise(target_future.dropna(), construction=construction).reindex(target.index)
        return log_target_future - log_target
    if construction == "average_log_growth_1_to_h":
        log_target_future = _log_or_raise(target_future.dropna(), construction=construction).reindex(target.index)
        return (log_target_future - log_target) / horizon
    raise AssertionError(f"unhandled horizon_target_construction={construction!r}")


def _is_average_construction(construction: str) -> bool:
    return construction in {
        "average_growth_1_to_h",
        "average_difference_1_to_h",
        "average_log_growth_1_to_h",
    }


def inverse_horizon_target(
    y_hat: float,
    y_anchor: float,
    construction: str,
    *,
    horizon: int = 1,
) -> float:
    """Convert a forecast on construction scale back to raw target level."""
    horizon = _positive_horizon(horizon)
    construction = canonicalize_horizon_target_construction(construction)
    if construction not in OPERATIONAL_CONSTRUCTIONS:
        raise ValueError(
            f"unknown horizon_target_construction={construction!r}; "
            f"operational set is {sorted(SUPPORTED_CONSTRUCTIONS)}"
        )
    y_hat_f = float(y_hat)
    if construction == "future_target_level_t_plus_h":
        return y_hat_f
    if construction == "future_diff":
        return float(y_anchor) + y_hat_f
    if construction == "average_difference_1_to_h":
        return float(y_anchor) + y_hat_f * horizon
    if construction == "average_growth_1_to_h":
        y_anchor_f = _require_positive_scalar(y_anchor, construction=construction, name="target_anchor")
        return y_anchor_f * (1.0 + y_hat_f * horizon)
    y_anchor_f = _require_positive_scalar(y_anchor, construction=construction, name="target_anchor")
    exponent = y_hat_f * horizon if _is_average_construction(construction) else y_hat_f
    return y_anchor_f * float(np.exp(exponent))


def is_log_space(construction: str) -> bool:
    """True if the forecast scale is logarithmic."""
    return canonicalize_horizon_target_construction(construction) in {
        "future_logdiff",
        "average_log_growth_1_to_h",
    }


def construction_scale(construction: str) -> str:
    """Return a stable label for the target-construction scale."""
    construction = canonicalize_horizon_target_construction(construction)
    if construction == "future_target_level_t_plus_h":
        return "level"
    if construction == "future_diff":
        return "difference"
    if construction == "future_logdiff":
        return "log_difference"
    if construction == "average_growth_1_to_h":
        return "average_growth"
    if construction == "average_difference_1_to_h":
        return "average_difference"
    if construction == "average_log_growth_1_to_h":
        return "average_log_growth"
    raise ValueError(
        f"unknown horizon_target_construction={construction!r}; "
        f"operational set is {sorted(SUPPORTED_CONSTRUCTIONS)}"
    )


def forward_scalar(y_val: float, y_anchor: float, construction: str, *, horizon: int = 1) -> float:
    """Apply forward transform to a single scalar forecast or actual value."""
    horizon = _positive_horizon(horizon)
    construction = canonicalize_horizon_target_construction(construction)
    if construction not in OPERATIONAL_CONSTRUCTIONS:
        raise ValueError(
            f"unknown horizon_target_construction={construction!r}; "
            f"operational set is {sorted(SUPPORTED_CONSTRUCTIONS)}"
        )
    y_val_f = float(y_val)
    if construction == "future_target_level_t_plus_h":
        return y_val_f
    if construction == "future_diff":
        return y_val_f - float(y_anchor)
    if construction == "average_difference_1_to_h":
        return (y_val_f - float(y_anchor)) / horizon
    y_anchor_f = _require_positive_scalar(y_anchor, construction=construction, name="target_anchor")
    y_val_f = _require_positive_scalar(y_val_f, construction=construction, name="target")
    if construction == "average_growth_1_to_h":
        return (y_val_f / y_anchor_f - 1.0) / horizon
    log_diff = float(np.log(y_val_f) - np.log(y_anchor_f))
    if construction == "average_log_growth_1_to_h":
        return log_diff / horizon
    return log_diff
