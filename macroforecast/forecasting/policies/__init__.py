"""Forecast-policy strategies for the runner (Phase 3 of the decomposition).

Every forecast policy is an equal strategy behind one registry: the direct
policy is registered exactly like recursive/path_average instead of living
inline in the dispatcher (the historical asymmetry of
``runner._fit_predict_origin``). ``dispatch`` preserves that dispatcher's
semantics bit-for-bit: ``path_average`` and ``recursive`` route to their
strategies, and everything else -- ``direct``, ``direct_average``, or any
other value -- runs the direct body.

The panel strategy is registered for symmetry and discoverability, but it has
a different call contract (canonical panel + per-run keyword arguments, see
``policies.panel``) and is invoked by the panel runner, never by ``dispatch``.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from macroforecast.forecasting.policies.base import (  # noqa: F401  (re-export)
    _FitOutcome,
    _OriginRunConfig,
    _fit_one_model_at_origin,
)
from macroforecast.forecasting.policies.direct import forecast_direct_origin
from macroforecast.forecasting.policies.panel import forecast_panel_origin
from macroforecast.forecasting.policies.path_average import forecast_path_average_origin
from macroforecast.forecasting.policies.recursive import forecast_recursive_origin

POLICY_FORECASTERS: dict[str, Callable[..., list[dict[str, Any]]]] = {
    "direct": forecast_direct_origin,
    "direct_average": forecast_direct_origin,
    "path_average": forecast_path_average_origin,
    "recursive": forecast_recursive_origin,
    "panel": forecast_panel_origin,
}

# The feature-matrix policies share the per-origin ``(item, cfg)`` contract and
# may be routed by ``dispatch``; the panel strategy may not (different input
# contract), which is why ``dispatch`` never falls through to it.
_FEATURE_MATRIX_POLICIES = frozenset(
    {"direct", "direct_average", "path_average", "recursive"}
)


def dispatch(
    item: Mapping[str, Any],
    cfg: _OriginRunConfig,
) -> list[dict[str, Any]]:
    """Route one per-origin feature-matrix item to its policy strategy.

    Exactly the historical ``runner._fit_predict_origin`` semantics: an item
    whose ``forecast_policy`` is ``path_average`` or ``recursive`` runs that
    strategy; anything else (``direct``, ``direct_average``, a missing key, or
    an unknown value) runs the direct body.
    """

    policy = item.get("forecast_policy")
    key = str(policy) if policy is not None else "direct"
    if key not in _FEATURE_MATRIX_POLICIES:
        key = "direct"
    return POLICY_FORECASTERS[key](item, cfg)
