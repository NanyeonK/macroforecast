"""Per-origin feature-builder fit sharing for the forecasting runner (Gap A).

Promotes the per-origin fitted feature builder (the ``FeatureSpec.fit()``
result -- e.g. the PCA/MARX/SIR numerical state) into the SAME shared
cross-arm cache dict that ``preprocessing_stage.py`` already uses for the
per-origin ``FittedPreprocessor``/``_PreparedStage`` tiers. Before this module,
the fitted feature builder lived in a run()-LOCAL variable
(``fitted_feature_cache``), so two arms that differ only in ``model`` -- the
single most common pipeline comparison -- each refit the (often expensive)
feature transform at every origin. This module lets such arms compute the fit
exactly ONCE per origin in the serial path.

Key design (investigated empirically against ``window/core.py`` and
``feature_engineering/specs.py``):

* ``FeatureSpec.to_dict()`` already serializes EVERY semantic field, including
  ``horizon``/``horizons``/``target_mode``/``target_transform``/``feature_steps``.
  ``forecasting.policy_config._feature_spec_for_policy`` bakes the resolved
  horizon and forecast-policy-derived ``target_mode``/``target_transform`` into a
  fresh ``FeatureSpec`` instance per (horizon, policy) call, so hashing
  ``to_dict()`` automatically gives distinct keys across horizons/policies
  wherever the FIT could actually depend on them (e.g. a supervised
  ``sliced_inverse_regression`` feature step consumes the h-shifted,
  policy-transformed target). The one exception is ``forecast_policy="recursive"``,
  which always pins ``FeatureSpec.horizon=1`` regardless of the outer requested
  horizon -- but ``window/core.py``'s non-panel (``exclude_origin=False``)
  ``estimation_end_pos``/``fit_end_pos`` derivation does not depend on
  ``test.horizon`` either, so the recursive feature fit genuinely IS
  horizon-independent and sharing it across horizons (via the cache already
  being threaded across horizons by ``_run_multiple_horizons``) is correct, not
  an accident.
* The window/stage-policy machinery is NOT reflected in ``FeatureSpec.to_dict()``
  at all (it lives on a separate ``StagePolicy`` object and is never passed to
  ``FeatureSpec.fit()``), so two arms could have identical ``features`` content
  but a different sample of rows actually fed to ``.fit()`` -- e.g. a per-arm
  ``window`` override (a real, tested configuration; see
  ``tests/pipeline/test_per_arm_window.py``) or a different ``feature_policy``
  scope. Trusting content-digest + ``origin_pos`` alone would risk a silent
  wrong-share in that case. Instead the key ALSO carries the exact
  ``(fit_start_pos, fit_end_pos)`` / ``(estimation_start_pos, estimation_end_pos)``
  integer row-position bounds that ``window/core.py`` already computes for this
  origin -- byte-identity of those bounds is the arbiter, so the key is
  self-verifying no matter WHY two arms' sample might differ (window override,
  embargo, retrain cadence, feature-policy scope), independent of any
  eligibility gating done by callers (e.g. ``pipeline/run.py``).
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from macroforecast.feature_engineering import FeatureSpec, FittedFeatureBuilder
from macroforecast.window import StagePolicy


def _feature_fit_sample_bounds(
    item: Mapping[str, Any], feature_stage_policy: StagePolicy
) -> tuple[str, int, int] | None:
    """The exact (scope, start_pos, end_pos) row bounds fed to ``.fit()`` here.

    Returns ``None`` when the bounds cannot be resolved robustly (a ``custom``
    stage-policy selector, or a missing position field) -- callers must treat a
    ``None`` result as "do not share", never as a wildcard match.
    """
    row = item.get("row", {})
    if not isinstance(row, Mapping):
        return None
    scope = feature_stage_policy.scope
    if scope == "fit_window":
        start, end = row.get("fit_start_pos"), row.get("fit_end_pos")
    elif scope == "origin_available":
        start, end = row.get("estimation_start_pos"), row.get("estimation_end_pos")
    else:
        # "custom" (arbitrary selector) has no stable integer bounds; "full_panel"
        # and "fixed_reference" are fit once before the per-origin loop entirely
        # (see ``fixed_feature_builder`` in runner.py) and never reach this helper.
        return None
    if start is None or end is None:
        return None
    return (scope, int(start), int(end))


def _feature_content_digest(features: FeatureSpec, feature_stage_policy: StagePolicy) -> str:
    """Stable SHA-256 digest over the feature spec + feature stage policy content.

    Mirrors ``PreprocessorStore.key``'s convention (``json.dumps(...,
    sort_keys=True, default=str)`` then hash) for consistency with the existing
    on-disk preprocessing cache key. ``default=str`` covers any non-JSON-native
    value (callables in ``StagePolicy.selector``/``metadata``, numpy scalars).
    """
    payload = {
        "features": features.to_dict(),
        "feature_stage_policy": feature_stage_policy.to_dict(),
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _feature_cache_key(
    features: FeatureSpec,
    feature_stage_policy: StagePolicy,
    item: Mapping[str, Any],
    *,
    vintage_id: Any | None = None,
) -> tuple[Any, ...] | None:
    """Namespaced, content-addressed key for the shared feature-fit cache.

    Tagged ``"features"`` so it can never collide with the preprocessing cache's
    own ``("origin_pos", N)`` / ``("prepared", ...)`` / ``("prepared_base", ...)``
    / ``("estimation_span", ...)`` keys in the same shared dict. Returns ``None``
    when the fit sample bounds cannot be resolved robustly (see
    ``_feature_fit_sample_bounds``) -- callers must skip cache read/write
    entirely in that case rather than share under an ambiguous key.
    """
    bounds = _feature_fit_sample_bounds(item, feature_stage_policy)
    if bounds is None:
        return None
    digest = _feature_content_digest(features, feature_stage_policy)
    key = ("features", digest, bounds)
    if vintage_id is None:
        return key
    return key + ("vintage", vintage_id)


def _fitted_feature_builder_for_origin(
    features: FeatureSpec,
    feature_fit_panel: Any,
    *,
    prepared_metadata: Mapping[str, Any] | None,
    feature_stage_policy: StagePolicy,
    item: Mapping[str, Any],
    preprocessing_cache: dict[Any, Any] | None,
    vintage_id: Any | None = None,
) -> FittedFeatureBuilder:
    """Fit the per-origin feature builder, sharing it across arms when possible.

    Called ONLY at the moment a fresh fit is actually due (the caller's own
    retrain-cadence check, e.g. ``update="never"``/``"on_retrain"``/an integer
    interval, has already decided to refit) -- so this function does not
    duplicate or alter that cadence decision, it only decides HOW the fit that
    is about to happen gets computed: from the shared cache when a
    byte-identical fit was already computed for another arm at this exact
    sample, or freshly via ``features.fit()`` otherwise (which then populates
    the cache for the next arm). When ``preprocessing_cache`` is ``None`` (the
    parallel backend, or a direct ``run()`` caller that opts out) this is
    exactly ``features.fit(feature_fit_panel, metadata=prepared_metadata)`` --
    byte-for-byte the pre-existing behavior.
    """
    cache_key = (
        _feature_cache_key(
            features,
            feature_stage_policy,
            item,
            vintage_id=vintage_id,
        )
        if preprocessing_cache is not None
        else None
    )
    if cache_key is not None:
        cached = preprocessing_cache.get(cache_key)  # type: ignore[union-attr]
        if isinstance(cached, FittedFeatureBuilder):
            return cached
    fitted = features.fit(feature_fit_panel, metadata=prepared_metadata)
    if cache_key is not None:
        preprocessing_cache[cache_key] = fitted  # type: ignore[index]
    return fitted


__all__ = [
    "_feature_cache_key",
    "_feature_content_digest",
    "_feature_fit_sample_bounds",
    "_fitted_feature_builder_for_origin",
]
