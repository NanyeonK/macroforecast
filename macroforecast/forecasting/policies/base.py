"""Shared policy machinery: the per-origin run configuration and the
select -> fit -> predict -> save skeleton every feature-matrix policy runs
through (Phase 3 of the runner decomposition; bodies moved verbatim from
``macroforecast.forecasting.runner``).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from macroforecast.model_selection import (
    SearchSpec,
    select_by_information_criterion,
    select_params,
)
from macroforecast.window import StagePolicy

from macroforecast.forecasting.model_resolution import (
    _ModelRun,
    _actual_model_params,
)
from macroforecast.forecasting.selection_stage import (
    _SELECTION_TUNED_KEY,
    _resolve_degraded_selection,
    _selection_for_model,
)


@dataclass(frozen=True)
class _OriginRunConfig:
    """The per-origin run configuration shared by every feature-matrix forecast
    policy (direct / direct_average / path_average). Bundles the ten arguments that
    were previously threaded as keyword parameters through each ``_fit_predict_*``
    function and both call sites. ``param_cache`` and ``selection_cache`` are shared
    mutable dicts (mutated in place across origins); the dataclass only freezes the
    references, not their contents.
    """

    model_runs: list[_ModelRun]
    selection: "SearchSpec | Mapping[str, SearchSpec | None] | None"
    selection_policy: StagePolicy
    selection_metric: "str | Callable[..., float]"
    maximize_selection: bool
    param_cache: dict[str, dict[str, Any]]
    selection_cache: dict[str, Any]
    selection_random_state: int | None
    save_models: bool
    model_store: "str | Path"


@dataclass(frozen=True)
class _FitOutcome:
    """What one model produced at one origin via the shared skeleton.

    ``prediction`` is ``None`` when the caller supplied no ``X_test`` (the
    recursive policy predicts by rolling its own panel forward instead of a
    single ``fit.predict`` call).
    """

    fit: Any
    fit_params: dict[str, Any]
    selection_metadata: dict[str, Any] | None
    stored_model: dict[str, Any] | None
    prediction: pd.Series | None


def _fit_one_model_at_origin(
    model_run: _ModelRun,
    X_fit: pd.DataFrame,
    y_fit: Any,
    X_sel: pd.DataFrame,
    y_sel: Any,
    X_test: pd.DataFrame | None,
    cfg: _OriginRunConfig,
    *,
    direct_capable_flag: bool,
    cache_key: str,
    row: Mapping[str, Any],
    selection_splits: Sequence[Any] | None,
    target_step: int,
    allow_non_temporal_splits: bool,
    ic_selection_enabled: bool = True,
    degraded_alias: str | None = None,
    retuned_metadata_extras: Mapping[str, Any] | None = None,
    reused_metadata_extras: Mapping[str, Any] | None = None,
) -> _FitOutcome:
    """The select -> fit -> predict -> save skeleton shared by the feature-matrix
    policies (the direct/direct_average inline body, recursive, and each
    path_average step). One implementation of the block that used to be
    copy-pasted per policy, so "one policy bug = four policy bugs" cannot recur:
    the IC-vs-CV selection branch, graceful selection degradation, the
    ``direct=True`` injection for direct-capable iterated models, the
    param/selection cache handling (the caller owns the CACHE KEY -- per-target
    for direct/recursive, per-step for path_average), and model saving all live
    here. The panel policy is NOT forced through this skeleton: its input
    contract (DataBundle panel, not an engineered feature matrix) is different.

    Variation points, all keyword-only:

    - ``direct_capable_flag``: inject ``direct=True`` into the fit call and IC
      selection (the direct policy and path_average per-step projections; the
      recursive roll-forward never injects it).
    - ``ic_selection_enabled``: the recursive policy historically routes
      IC-capable models through the CV/validation-split branch; ``False``
      preserves that.
    - ``degraded_alias``: display alias for the degraded-selection warning
      (path steps append " (path step N)").
    - ``retuned_metadata_extras`` / ``reused_metadata_extras``: policy-specific
      keys appended to the selection metadata when freshly tuned vs when reused
      from cache or degraded (recursive tags retuned metadata with
      ``recursive``/``future_feature_policy``; path tags everything with
      ``path_step``).
    """

    model_spec = model_run.spec
    retune = bool(row.get("retune", True))
    ic_fixed_params = {"direct": True} if direct_capable_flag else None
    selected, use_model_default_selection = _selection_for_model(
        cfg.selection, model_run
    )
    should_select = selected is not None or (
        use_model_default_selection and bool(model_spec.search_spaces)
    )
    selection_metadata: dict[str, Any] | None = None
    uses_ic = ic_selection_enabled and str(
        getattr(model_spec, "selection_method", "cv")
    ).lower() in ("bic", "aic", "aicc")
    if should_select and uses_ic:
        # Information-criterion models (AR, FM) select their order by BIC/AIC on
        # the full training sample, so they need no validation split. This is
        # both paper-faithful and what lets the autoregression run with a window
        # that carries no validation block; for path_average steps it is also
        # what preserves the horizon-1 direct==path invariant (a CV split would
        # score the order on a truncated sample and pick a different order).
        if retune or cache_key not in cfg.param_cache:
            result = select_by_information_criterion(
                model_spec,
                X_sel,
                y_sel,
                search=selected,
                criterion=str(model_spec.selection_method).lower(),
                fixed_params=ic_fixed_params,
            )
            cfg.param_cache[cache_key] = dict(result.best_params)
            selection_metadata = {
                **result.to_metadata(),
                "policy": cfg.selection_policy.to_dict(),
                "retuned": True,
                **(retuned_metadata_extras or {}),
            }
            cfg.selection_cache[cache_key] = selection_metadata
            cfg.selection_cache[_SELECTION_TUNED_KEY] = True
            best_params = dict(cfg.param_cache.get(cache_key, {}))
        else:
            selection_metadata = cfg.selection_cache.get(cache_key)
            if selection_metadata is not None:
                selection_metadata = {
                    **selection_metadata,
                    "retuned": False,
                    **(reused_metadata_extras or {}),
                }
            best_params = dict(cfg.param_cache.get(cache_key, {}))
    elif should_select:
        if not selection_splits:
            best_params, selection_metadata = _resolve_degraded_selection(
                cache_key=cache_key,
                alias=degraded_alias if degraded_alias is not None else model_run.alias,
                target_step=target_step,
                origin_label=row.get("origin"),
                param_cache=cfg.param_cache,
                selection_cache=cfg.selection_cache,
                selection_policy=cfg.selection_policy,
            )
            if reused_metadata_extras and isinstance(selection_metadata, dict):
                selection_metadata = {**selection_metadata, **reused_metadata_extras}
        elif retune or cache_key not in cfg.param_cache:
            result = select_params(
                model_spec,
                X_sel,
                y_sel,
                search=selected,
                splits=selection_splits,
                metric=cfg.selection_metric,
                maximize=cfg.maximize_selection,
                random_state=cfg.selection_random_state if selected is None else None,
                allow_non_temporal_splits=allow_non_temporal_splits,
            )
            cfg.param_cache[cache_key] = dict(result.best_params)
            selection_metadata = {
                **result.to_metadata(),
                "policy": cfg.selection_policy.to_dict(),
                "retuned": True,
                **(retuned_metadata_extras or {}),
            }
            cfg.selection_cache[cache_key] = selection_metadata
            cfg.selection_cache[_SELECTION_TUNED_KEY] = True
            best_params = dict(cfg.param_cache.get(cache_key, {}))
        else:
            selection_metadata = cfg.selection_cache.get(cache_key)
            if selection_metadata is not None:
                selection_metadata = {
                    **selection_metadata,
                    "retuned": False,
                    **(reused_metadata_extras or {}),
                }
            best_params = dict(cfg.param_cache.get(cache_key, {}))
    else:
        best_params = {}
    call_params = (
        {**best_params, "direct": True} if direct_capable_flag else best_params
    )
    fit_params = _actual_model_params(model_spec, call_params)
    fit = model_spec(X_fit, y_fit, **call_params)
    stored_model = (
        _store_model_fit(
            fit,
            root=cfg.model_store,
            alias=model_run.alias,
            model_spec=model_spec,
            row=row,
            params=fit_params,
            selection_metadata=selection_metadata,
        )
        if cfg.save_models
        else None
    )
    prediction = (
        _prediction_series(fit.predict(X_test), index=X_test.index)
        if X_test is not None
        else None
    )
    return _FitOutcome(
        fit=fit,
        fit_params=fit_params,
        selection_metadata=selection_metadata,
        stored_model=stored_model,
        prediction=prediction,
    )


# Imported at the bottom: these helpers still live in runner.py, and runner.py
# imports this package after defining them, so a top-of-module import would be
# circular.
from macroforecast.forecasting.runner import (  # noqa: E402
    _prediction_series,
    _store_model_fit,
)
