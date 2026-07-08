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

import numpy as np
import pandas as pd

from macroforecast.models import ModelSpec, save_fit
from macroforecast.meta.config import _derive_random_state

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
    model_random_seed: int | None
    model_random_alias: str | None
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
    selected_method = (
        str(getattr(selected, "method", "")).lower().replace("-", "_")
        if selected is not None
        else ""
    )
    explicit_ic = selected_method in {"information_criterion", "ic"}
    model_owned_ic = str(
        getattr(model_spec, "selection_method", "cv")
    ).lower() in ("bic", "aic", "aicc")
    uses_ic = explicit_ic or (ic_selection_enabled and model_owned_ic)
    if should_select and uses_ic:
        # Information-criterion models (AR, FM) select their order by BIC/AIC on
        # the full training sample, so they need no validation split. This is
        # both paper-faithful and what lets the autoregression run with a window
        # that carries no validation block; for path_average steps it is also
        # what preserves the horizon-1 direct==path invariant (a CV split would
        # score the order on a truncated sample and pick a different order).
        if retune or cache_key not in cfg.param_cache:
            criterion = (
                str(selected.criterion).lower()
                if explicit_ic and selected is not None and selected.criterion is not None
                else "bic"
                if explicit_ic
                else str(model_spec.selection_method).lower()
            )
            result = select_by_information_criterion(
                model_spec,
                X_sel,
                y_sel,
                search=selected,
                criterion=criterion,
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
            splitter_override = (
                selected is not None and selected.validation_splitter is not None
            )
            result = select_params(
                model_spec,
                X_sel,
                y_sel,
                search=selected,
                splits=None if splitter_override else selection_splits,
                metric=cfg.selection_metric,
                maximize=cfg.maximize_selection,
                random_state=cfg.selection_random_state if selected is None else None,
                allow_non_temporal_splits=allow_non_temporal_splits
                or splitter_override,
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
    call_params = _with_derived_random_state(
        model_spec,
        call_params,
        seed=cfg.model_random_seed,
        alias=cfg.model_random_alias or model_run.alias,
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


def _with_derived_random_state(
    model_spec: ModelSpec,
    params: Mapping[str, Any],
    *,
    seed: int | None,
    alias: str,
) -> dict[str, Any]:
    """Inject a pipeline-derived random_state only when no explicit value exists."""

    resolved = dict(params)
    if (
        seed is None
        or "random_state" not in model_spec.default_params
        or "random_state" in model_spec.params
        or "random_state" in resolved
    ):
        return resolved
    derived = _derive_random_state(seed, alias)
    if derived is not None:
        resolved["random_state"] = derived
    return resolved


# ---------------------------------------------------------------------------
# Shared policy output machinery (Phase 5 of the runner decomposition; bodies
# moved verbatim from ``macroforecast.forecasting.runner``): prediction /
# variance / quantile coercion onto the test index, the model store, the
# per-model parameter-cache key, and origin -> target date mapping.
# ---------------------------------------------------------------------------


def _prediction_series(prediction: Any, *, index: pd.Index) -> pd.Series:
    if isinstance(prediction, pd.Series):
        return _aligned_or_positional_series(
            prediction,
            index=index,
            label="model prediction",
        )
    if isinstance(prediction, pd.DataFrame):
        if prediction.shape[1] != 1:
            raise ValueError("model prediction DataFrame must have exactly one column")
        return _aligned_or_positional_series(
            prediction.iloc[:, 0],
            index=index,
            label="model prediction",
        )
    values = np.asarray(prediction).reshape(-1)
    if len(values) != len(index):
        raise ValueError("model prediction length does not match X_test")
    return pd.Series(values, index=index)


def _aligned_or_positional_series(
    values: pd.Series,
    *,
    index: pd.Index,
    label: str,
) -> pd.Series:
    if len(values) != len(index):
        raise ValueError(f"{label} length does not match X_test")
    if values.index.equals(index):
        return values.copy()
    if _is_default_position_index(values.index, len(index)):
        return pd.Series(values.to_numpy(), index=index, name=values.name)
    raise ValueError(
        f"{label} index does not match X_test. Return an array-like object for "
        "positional predictions, or return a pandas object indexed by X_test.index."
    )


def _is_default_position_index(index: pd.Index, n: int) -> bool:
    if isinstance(index, pd.RangeIndex):
        return index.equals(pd.RangeIndex(n))
    try:
        return bool(np.array_equal(index.to_numpy(dtype=int), np.arange(n)))
    except (TypeError, ValueError):
        return False


def _variance_series(
    fit: Any,
    *,
    X_test: pd.DataFrame | None = None,
    index: pd.Index,
) -> pd.Series | None:
    if not hasattr(fit, "predict_variance"):
        return None
    prediction = None
    if X_test is not None:
        try:
            prediction = fit.predict_variance(X_test)
        except TypeError:
            prediction = None
    try:
        if prediction is None:
            prediction = fit.predict_variance(horizon=len(index))
    except TypeError:
        prediction = fit.predict_variance(len(index))
    values = _positional_prediction_values(prediction, expected_len=len(index))
    return pd.Series(values, index=index, name="variance_prediction")


def _quantile_frame(
    fit: Any, *, X_test: pd.DataFrame, index: pd.Index
) -> pd.DataFrame | None:
    if not hasattr(fit, "predict_quantiles"):
        return None
    prediction = fit.predict_quantiles(X_test)
    if isinstance(prediction, pd.DataFrame):
        frame = prediction.copy()
        if len(frame) != len(index):
            raise ValueError("quantile prediction length does not match X_test")
        if frame.index.equals(index):
            return frame
        if _is_default_position_index(frame.index, len(index)):
            frame.index = index
            return frame
        raise ValueError(
            "quantile prediction index does not match X_test. Return a DataFrame "
            "indexed by X_test.index, use RangeIndex(len(X_test)), or return a mapping "
            "of array-like quantile predictions."
        )
    if isinstance(prediction, Mapping):
        columns: dict[str, np.ndarray] = {}
        for level, values in prediction.items():
            arr = np.asarray(values, dtype=float).reshape(-1)
            if len(arr) != len(index):
                raise ValueError("quantile prediction length does not match X_test")
            columns[str(level)] = arr
        return pd.DataFrame(columns, index=index)
    raise TypeError("quantile predictions must be a DataFrame or mapping")


def _store_model_fit(
    fit: Any,
    *,
    root: str | Path,
    alias: str,
    model_spec: ModelSpec,
    row: Mapping[str, Any],
    params: Mapping[str, Any],
    selection_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    model_dir = Path(root) / _safe_path_part(alias)
    model_dir.mkdir(parents=True, exist_ok=True)
    stem = _model_store_stem(row)
    pickle_path = model_dir / f"{stem}.pkl"
    metadata_path = model_dir / f"{stem}.json"
    metadata = {
        "alias": alias,
        "model": model_spec.name,
        "model_spec": model_spec.to_metadata(),
        "params": dict(params),
        "model_selection": selection_metadata,
        "window": dict(row),
    }
    return save_fit(
        fit,
        pickle_path,
        metadata_path=metadata_path,
        metadata=metadata,
    ).to_dict()


def _model_store_stem(row: Mapping[str, Any]) -> str:
    origin_pos = row.get("origin_pos", "unknown")
    horizon = row.get("horizon", "unknown")
    origin = row.get("origin")
    if isinstance(origin, pd.Timestamp):
        origin_label = origin.strftime("%Y%m%d")
    else:
        origin_label = str(origin).replace(" ", "_").replace(":", "-")
    target_key = row.get("target_key")
    suffix = "" if target_key is None else f"_{_safe_path_part(target_key)}"
    return f"origin_{origin_pos}_h{horizon}_{_safe_path_part(origin_label)}{suffix}"


def _model_cache_key(alias: str, target_key: Any | None) -> str:
    if target_key is None:
        return str(alias)
    return f"{alias}::{target_key}"


def _forecast_target_dates(
    index: pd.Index,
    *,
    base_index: pd.Index | None,
    horizon: int,
) -> pd.Series:
    if base_index is None:
        return pd.Series(index, index=index)
    positions = base_index.get_indexer(index)
    target_positions = positions + int(horizon)
    valid = (positions >= 0) & (target_positions < len(base_index))
    values = pd.Series(pd.NA, index=index, dtype="object")
    if valid.any():
        valid_positions = target_positions[valid]
        values.iloc[np.flatnonzero(valid)] = list(base_index[valid_positions])
    return values



def _safe_path_part(value: Any) -> str:
    text = str(value)
    keep = [char if char.isalnum() or char in {"-", "_", "."} else "_" for char in text]
    out = "".join(keep).strip("._")
    return out or "model"



def _positional_prediction_values(prediction: Any, *, expected_len: int) -> np.ndarray:
    if isinstance(prediction, pd.DataFrame):
        if prediction.shape[1] != 1:
            raise ValueError(
                "variance prediction DataFrame must have exactly one column"
            )
        values = prediction.iloc[:, 0].to_numpy(dtype=float)
    elif isinstance(prediction, pd.Series):
        values = prediction.to_numpy(dtype=float)
    else:
        values = np.asarray(prediction, dtype=float).reshape(-1)
    if len(values) != expected_len:
        raise ValueError("variance prediction length does not match X_test")
    return values
