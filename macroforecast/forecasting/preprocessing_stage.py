"""Per-origin panel preparation for the forecasting runner (Phase 4 of the
runner decomposition; bodies moved verbatim from
``macroforecast.forecasting.runner``). Holds ``_prepare_origin_panel`` with its
three cache tiers (per-origin FittedPreprocessor, cross-arm prepared stage,
cross-horizon base transform) and ALL cache-key logic -- the key structure is
moved wholesale and must not be touched (plan Phase 4 note).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from macroforecast.data import DataBundle, spec as data_spec
from macroforecast.feature_engineering import FeatureSpec, FittedFeatureBuilder
from macroforecast.preprocessing import FittedPreprocessor, PreprocessSpec
from macroforecast.preprocessing.cache import PreprocessorStore
from macroforecast.window import StagePolicy, stage_panel


@dataclass(frozen=True)
class _PreparedStage:
    panel: pd.DataFrame
    fitted_preprocessing: FittedPreprocessor | None
    metadata: dict[str, Any] | None
    # Full per-origin panel metadata (the ``macroforecast_metadata`` payload that
    # ``_prepare_origin_panel`` would otherwise leave on ``panel.attrs``). It is
    # held here on the dataclass instead of on ``panel.attrs`` because the cached
    # ``panel`` is operated on repeatedly in the per-origin/per-arm hot loop, and
    # pandas deep-copies ``.attrs`` on every operation via ``__finalize__``. The
    # preprocessing metadata is large (EM/factor + standardization state, ~40 KB),
    # so carrying it on ``.attrs`` makes that deepcopy dominate runtime in
    # multi-horizon runs that share the panel across many origins. Consumers read
    # the metadata from this field and re-attach it only where it is needed.
    panel_metadata: dict[str, Any] | None = None


def _preprocessing_cache_key(
    item: Mapping[str, Any],
    *,
    vintage_id: Any | None = None,
) -> Any:
    """Horizon-independent key for the shared per-origin preprocessing cache.

    The spec-level EM/factor fit at one origin uses only the ``origin_available``
    (estimation) rows, which are fully determined by the origin POSITION and do not
    depend on the forecast horizon (test block / target row are excluded from the
    fit). Keying the cache on ``origin_pos`` alone therefore lets arms AND horizons
    of the same target reuse the identical FittedPreprocessor without recomputing
    the EM SVD, while staying leak-free (the fit never sees future rows).
    """
    row = item.get("row")
    if isinstance(row, Mapping) and "origin_pos" in row:
        key: Any = ("origin_pos", int(row["origin_pos"]))
        return _with_vintage_cache_tag(key, vintage_id)
    # Fallback: estimation block identity (still horizon-independent).
    est = item.get("estimation_idx")
    if est is not None:
        arr = np.asarray(est, dtype=int)
        key = ("estimation_span", int(arr[0]), int(arr[-1])) if arr.size else ("empty",)
        return _with_vintage_cache_tag(key, vintage_id)
    return _with_vintage_cache_tag(("origin_pos", item.get("origin_pos")), vintage_id)


def _with_vintage_cache_tag(cache_key: Any, vintage_id: Any | None) -> Any:
    if vintage_id is None:
        return cache_key
    return tuple(cache_key) + ("vintage", vintage_id)


def _origin_pos_for_store_key(cache_key: Any, item: Mapping[str, Any]) -> int | None:
    """Resolve the integer ``origin_pos`` used in the on-disk store key.

    The on-disk store keys on the EXACT ``(spec, target, origin_pos)`` triple, so
    it needs a stable integer origin position. We take it from the same source the
    in-memory key uses -- the origin row's ``origin_pos`` (preferred) -- so the two
    tiers agree on what "the same origin" means. If no integer origin position is
    available (only the ``estimation_span``/``empty`` fallback cache keys), we
    return ``None`` and the caller skips the disk tier entirely, falling back to a
    plain (correct) recompute rather than risk an ambiguous key.
    """

    row = item.get("row")
    if isinstance(row, Mapping):
        pos = row.get("origin_pos")
        if pos is not None:
            return int(pos)
    # Mirror the in-memory key's primary form: ("origin_pos", N).
    if (
        isinstance(cache_key, tuple)
        and len(cache_key) == 2
        and cache_key[0] == "origin_pos"
        and cache_key[1] is not None
    ):
        return int(cache_key[1])
    return None


def _prepare_origin_panel(
    panel: pd.DataFrame,
    *,
    features: FeatureSpec | None,
    preprocessing: PreprocessSpec | None,
    preprocessing_policy: StagePolicy | None,
    item: dict[str, Any],
    include_target_pos: bool = True,
    fitted_preprocessing: FittedPreprocessor | None = None,
    preprocessing_cache: dict[Any, FittedPreprocessor | _PreparedStage | FittedFeatureBuilder] | None = None,
    cache_key: Any | None = None,
    preprocessing_store: PreprocessorStore | None = None,
    target: str | None = None,
) -> _PreparedStage:
    if preprocessing is None or preprocessing_policy is None:
        return _PreparedStage(
            panel=panel,
            fitted_preprocessing=None,
            metadata=None,
            panel_metadata=dict(panel.attrs.get("macroforecast_metadata", {})),
        )
    # Cross-arm reuse of the TRANSFORMED panel. The leak-free origin_available
    # transform re-runs the (expensive) EM imputation on the apply window, and the
    # output panel depends only on (panel, the origin's apply/available labels,
    # preprocessing spec) -- NOT on the arm's features/model. Arms of the same
    # target therefore produce an identical prepared panel at each origin, so the
    # first arm computes it and the rest reuse the cached _PreparedStage. This
    # removes the dominant per-arm EM-transform redundancy (the fit cache alone
    # only saved the cheaper .fit(), not the per-origin transform).
    prepared_key = (
        ("prepared",) + tuple(cache_key) + (int(bool(include_target_pos)),
            int(_origin_target_pos(panel.index, item)))
        if (preprocessing_cache is not None and cache_key is not None)
        else None
    )
    if (
        prepared_key is not None
        and preprocessing_cache is not None
        and prepared_key in preprocessing_cache
    ):
        cached = preprocessing_cache[prepared_key]
        if isinstance(cached, _PreparedStage):
            return cached
    # Cross-arm/cross-horizon reuse of the per-origin FittedPreprocessor (the EM
    # FIT). The fit depends on target+horizons metadata only, not the arm's
    # predictors/model, and is horizon-independent (origin_available rows), so the
    # first caller fits and the rest reuse via the origin-keyed cache.
    if (
        fitted_preprocessing is None
        and preprocessing_cache is not None
        and cache_key is not None
        and cache_key in preprocessing_cache
    ):
        candidate = preprocessing_cache[cache_key]
        if isinstance(candidate, FittedPreprocessor):
            fitted_preprocessing = candidate
    # Second cache tier: the on-disk content-addressed store. Consulted ONLY on an
    # in-memory miss, and ONLY when a store was passed (store is None -> this whole
    # block is skipped and behaviour is byte-identical to the no-store path). The
    # store key is the EXACT (PreprocessSpec, target, origin_pos) triple that
    # identifies this fit, so a loaded preprocessor is reused only for an identical
    # spec+target+origin -- never served for a different one. A store hit is also
    # written back into the in-memory dict so later same-process lookups stay fast.
    # NOTE: the store key encodes (PreprocessSpec, target, origin_pos) but NOT
    # ``preprocessing_policy.scope`` (origin_available vs fit_window). Within a
    # single run() the scope is constant, so reuse is safe; do NOT share one store
    # directory across runs that use a different scope for the same spec (a
    # fit_window fit would be served where an origin_available fit is expected).
    store_key: str | None = None
    store_origin_pos: int | None = None
    if (
        fitted_preprocessing is None
        and preprocessing_store is not None
        and cache_key is not None
        and target is not None
    ):
        store_origin_pos = _origin_pos_for_store_key(cache_key, item)
        if store_origin_pos is not None:
            store_key = preprocessing_store.key(
                preprocessing,
                target=str(target),
                origin_pos=store_origin_pos,
            )
            loaded = preprocessing_store.get(store_key)
            if isinstance(loaded, FittedPreprocessor):
                fitted_preprocessing = loaded
                if preprocessing_cache is not None:
                    preprocessing_cache[cache_key] = loaded
    if fitted_preprocessing is None:
        fit_panel = stage_panel(panel, item, preprocessing_policy)
        fit_policy = (
            "fit_window"
            if preprocessing_policy.scope in {"fit_window", "fixed_reference"}
            else "origin_available"
        )
        fitted = preprocessing.fit(
            _preprocessor_fit_input(fit_panel, features),
            policy=fit_policy,
        )
        if preprocessing_cache is not None and cache_key is not None:
            preprocessing_cache[cache_key] = fitted
        # Persist the freshly computed fit to the on-disk store so other
        # processes / later run() calls reuse it instead of recomputing.
        # (store_key is only set when preprocessing_store is not None; the explicit
        # check also narrows the Optional for the type checker.)
        if store_key is not None and preprocessing_store is not None:
            preprocessing_store.put(store_key, fitted)
    else:
        fitted = fitted_preprocessing
    apply_labels = _origin_apply_labels(
        panel.index,
        item,
        include_target_pos=include_target_pos,
    )
    cols = fitted.fit_panel.columns
    if fitted.preprocessing_scope == "fit_window":
        apply_panel = panel.reindex(apply_labels).loc[:, cols]
        transformed = fitted.transform(
            apply_panel, history=fitted.fit_panel, policy="fit_window"
        )
        prepared_panel = transformed.panel
    else:
        # Rows observable AT the forecast origin: every apply row whose position
        # is <= the origin position. This must exclude the ENTIRE forward test
        # block (origin+1 .. origin+horizon-1) and the appended target row, not
        # just the target row -- test_idx spans the whole horizon block, so a
        # naive estimation/fit/test union would still feed h-1 strictly-future
        # rows into the data-dependent preprocessing (EM imputation / outlier
        # statistics) fit and leak the future into training-row features.
        available_labels = _origin_available_labels(panel.index, item)
        # Cross-HORIZON reuse of the (dominant-cost) origin_available transform. The
        # transform of the rows observable at the origin (<= origin_pos) is
        # horizon-independent -- it depends only on (fitted, available), never on the
        # forecast horizon -- so cache that heavy base-panel transform under an
        # origin-keyed key and reuse it across every horizon at this origin. Only the
        # tiny horizon-specific forward/target rows (> origin_pos) are transformed per
        # call; passing history=fit_panel lets their first-difference / lag t-codes see
        # the origin row exactly as the whole-window transform would. Numerical
        # identity vs the un-split whole-window path is pinned by the serial==parallel
        # golden -- the parallel backend (preprocessing_cache=None) keeps that path.
        base_key = ("prepared_base",) + tuple(cache_key) if cache_key is not None else None
        if base_key is None:
            apply_panel = panel.reindex(apply_labels).loc[:, cols]
            transformed = fitted.transform(
                apply_panel,
                history=fitted.fit_panel,
                policy="origin_available",
                available=available_labels,
            )
            prepared_panel = transformed.panel
        else:
            base_panel = (
                preprocessing_cache.get(base_key)
                if preprocessing_cache is not None
                else None
            )
            if not isinstance(base_panel, pd.DataFrame):
                base_store_key = (
                    preprocessing_store.frame_key(
                        preprocessing,
                        target=str(target),
                        cache_key=cache_key,
                        kind="prepared_base",
                    )
                    if preprocessing_store is not None and target is not None
                    else None
                )
                if base_store_key is not None and preprocessing_store is not None:
                    loaded_base = preprocessing_store.get_frame(base_store_key)
                    if isinstance(loaded_base, pd.DataFrame):
                        base_panel = loaded_base
                if not isinstance(base_panel, pd.DataFrame):
                    base_panel = fitted.transform(
                        panel.reindex(available_labels).loc[:, cols],
                        history=fitted.fit_panel,
                        policy="origin_available",
                        available=available_labels,
                    ).panel
                    if base_store_key is not None and preprocessing_store is not None:
                        preprocessing_store.put_frame(base_store_key, base_panel)
                if preprocessing_cache is not None:
                    preprocessing_cache[base_key] = base_panel
            fwd_labels = apply_labels[~apply_labels.isin(available_labels)]
            if len(fwd_labels):
                fwd_panel = fitted.transform(
                    panel.reindex(fwd_labels).loc[:, cols],
                    history=fitted.fit_panel,
                    policy="origin_available",
                    available=available_labels,
                ).panel
                prepared_panel = pd.concat([base_panel, fwd_panel]).reindex(apply_labels)
            else:
                prepared_panel = base_panel.reindex(apply_labels)
            # concat/reindex drop the (horizon-independent) preprocessing metadata that
            # _detach_panel_metadata expects on ``.attrs``; restore it from the base.
            prepared_panel.attrs = dict(base_panel.attrs)
    # Move the heavy ``macroforecast_metadata`` payload off ``panel.attrs`` and
    # onto the dataclass. The transformed panel below is reindexed/dropna'd/fed to
    # feature fit+transform once per origin AND reused across arms/horizons from
    # the shared cache; pandas deep-copies ``.attrs`` on each of those operations
    # via ``__finalize__``, so leaving the ~40 KB preprocessing/standardization
    # metadata on ``.attrs`` makes that deepcopy dominate multi-horizon runtime.
    # Small keys that downstream code reads directly off the panel (e.g. the
    # transform-code map) are preserved on ``.attrs``; the full metadata is kept
    # on ``panel_metadata`` and re-attached by callers only where it is consumed.
    panel_metadata = _detach_panel_metadata(prepared_panel)
    stage = _PreparedStage(
        panel=prepared_panel,
        fitted_preprocessing=fitted,
        metadata=fitted.to_metadata(),
        panel_metadata=panel_metadata,
    )
    if prepared_key is not None and preprocessing_cache is not None:
        preprocessing_cache[prepared_key] = stage
    return stage


# Small ``.attrs`` keys that are cheap to deep-copy and are read directly off the
# panel by downstream stages (transform-code resolution, etc.). These stay on the
# panel; the large ``macroforecast_metadata`` payload is detached.
_LIGHT_PANEL_ATTR_KEYS = ("macroforecast_transform_codes",)


def _detach_panel_metadata(panel: pd.DataFrame) -> dict[str, Any] | None:
    """Strip the heavy ``macroforecast_metadata`` from ``panel.attrs`` in place.

    Returns the detached metadata dict (or ``None`` if absent). After this call
    ``panel.attrs`` retains only the small, deep-copy-cheap keys in
    ``_LIGHT_PANEL_ATTR_KEYS``, so the per-origin/per-arm pandas operations on the
    cached panel no longer pay the cost of deep-copying the full preprocessing
    metadata on every ``__finalize__``.
    """

    attrs = panel.attrs
    detached = attrs.get("macroforecast_metadata")
    light = {key: attrs[key] for key in _LIGHT_PANEL_ATTR_KEYS if key in attrs}
    panel.attrs = light
    return dict(detached) if isinstance(detached, Mapping) else detached


def _origin_target_pos(index: pd.Index, item: Mapping[str, Any]) -> int:
    """Position of the appended post-origin target row (origin_pos + horizon).

    Used only as part of the prepared-stage cache key so that two arms sharing the
    SAME (origin, horizon) reuse the identical transformed panel, while different
    horizons (different target rows) get distinct cache entries.
    """
    row = item.get("row", {})
    try:
        origin_pos = int(row.get("test_start_pos", item["test_idx"][0]))
        horizon = int(row.get("horizon", 1))
    except (TypeError, ValueError, IndexError, KeyError):
        return -1
    target_pos = origin_pos + horizon
    return target_pos if 0 <= target_pos < len(index) else -1


def _origin_apply_labels(
    index: pd.Index,
    item: dict[str, Any],
    *,
    include_target_pos: bool = True,
) -> pd.Index:
    positions = np.unique(
        np.concatenate([item["estimation_idx"], item["fit_idx"], item["test_idx"]])
    )
    if not include_target_pos:
        return index[positions]
    row = item.get("row", {})
    try:
        target_pos = int(row.get("test_start_pos", item["test_idx"][0])) + int(
            row.get("horizon", 1)
        )
    except (TypeError, ValueError, IndexError):
        target_pos = -1
    if 0 <= target_pos < len(index):
        positions = np.unique(np.concatenate([positions, np.asarray([target_pos])]))
    return index[positions]


def _origin_available_labels(index: pd.Index, item: dict[str, Any]) -> pd.Index:
    """Labels observable at the forecast origin: apply positions <= origin_pos.

    ``test_idx`` spans the whole forward horizon block for direct/path policies,
    so it contains rows strictly after the origin. Only rows at or before the
    origin position may inform the origin_available data-dependent preprocessing
    fit (imputation / outlier statistics); everything later is unrealised future.
    """
    positions = np.unique(
        np.concatenate([item["estimation_idx"], item["fit_idx"], item["test_idx"]])
    )
    row = item.get("row", {})
    try:
        origin_pos = int(row.get("test_start_pos", item["test_idx"][0]))
    except (TypeError, ValueError, IndexError):
        origin_pos = int(positions.max()) if len(positions) else -1
    available = positions[positions <= origin_pos]
    return index[available]


def _preprocessor_fit_input(fit_panel: pd.DataFrame, features: FeatureSpec | None) -> Any:
    if features is None:
        metadata = dict(fit_panel.attrs.get("macroforecast_metadata", {}))
        return DataBundle(fit_panel, metadata)
    target = features.target
    targets = features.targets or None
    if target is None and not targets:
        return fit_panel
    metadata = dict(fit_panel.attrs.get("macroforecast_metadata", {}))
    horizons: Any
    if features.horizons:
        horizons = features.horizons
    elif features.horizon is not None:
        horizons = features.horizon
    else:
        horizons = None
    # Preprocessing may create columns that the downstream FeatureSpec uses.
    # Validate target/horizon metadata here, but defer predictor validation until
    # after preprocessing has produced the panel consumed by feature engineering.
    return data_spec(
        DataBundle(fit_panel, metadata),
        target=target,
        targets=targets,
        horizons=horizons,
        predictors="all",
    )
