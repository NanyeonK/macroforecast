"""Target-availability-safe model selection for the forecasting runner
(Phase 4 of the runner decomposition; bodies moved verbatim from
``macroforecast.forecasting.runner``): availability filtering/masking, the
availability-safe validation splits, graceful selection degradation, split
re-mapping onto stage-local feature matrices, and per-model selection-spec
resolution.
"""
from __future__ import annotations

import warnings

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from macroforecast.model_selection import SearchSpec
from macroforecast.window import Split, StagePolicy, WindowSpec, make_splitter

from macroforecast.forecasting.model_resolution import _ModelRun, _run_keys


def _single_target(y: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(y, pd.Series):
        return y
    frame = pd.DataFrame(y)
    if frame.shape[1] != 1:
        raise ValueError(
            "forecasting runner currently expects exactly one target column"
        )
    return frame.iloc[:, 0].rename(str(frame.columns[0]))


def _filter_xy_to_target_availability(
    X: Any,
    y: Any,
    item: dict[str, Any],
    *,
    target_step: int,
) -> tuple[pd.DataFrame, pd.Series]:
    X_aligned, y_aligned = _align_feature_xy(X, y)
    mask = _target_availability_mask(
        X_aligned.index,
        item,
        target_step=target_step,
    )
    return X_aligned.loc[mask], y_aligned.loc[mask]


def _target_availability_mask(
    labels: pd.Index,
    item: dict[str, Any],
    *,
    target_step: int,
) -> np.ndarray:
    base_index = _target_availability_base_index(item, labels)
    positions = base_index.get_indexer(pd.Index(labels))
    if (positions < 0).any():
        missing = pd.Index(labels)[positions < 0]
        raise ValueError(
            "forecast target-availability filtering requires feature labels "
            f"to be contained in base_index; missing labels: {list(missing[:3])}"
        )
    origin_pos = int(item["row"]["origin_pos"])
    cutoff_pos = origin_pos - int(target_step)
    return positions <= cutoff_pos


def _target_availability_base_index(
    item: dict[str, Any],
    fallback: pd.Index,
) -> pd.Index:
    raw = item.get("base_index")
    if raw is None:
        return pd.Index(fallback)
    return pd.Index(raw)


def _target_availability_window_fields(
    item: dict[str, Any],
    *,
    target_step: int,
    policy: str = "row_pos_plus_target_step_lte_origin_pos",
) -> dict[str, Any]:
    origin_pos = int(item["row"]["origin_pos"])
    cutoff_pos = origin_pos - int(target_step)
    base_index = _target_availability_base_index(item, pd.Index([]))
    cutoff_label: Any | None = None
    if 0 <= cutoff_pos < len(base_index):
        cutoff_label = base_index[cutoff_pos]
    return {
        "target_availability_policy": policy,
        "target_availability_lag": int(target_step),
        "target_availability_end": cutoff_label,
        "target_availability_end_pos": int(cutoff_pos),
    }


_SELECTION_TUNED_KEY = "__macroforecast_selection_ever_tuned__"
_SELECTION_DEGRADED_KEY = "__macroforecast_selection_ever_degraded__"


def _resolve_degraded_selection(
    *,
    cache_key: str,
    alias: str,
    target_step: int,
    origin_label: Any,
    param_cache: dict[str, dict[str, Any]],
    selection_cache: dict[str, Any],
    selection_policy: StagePolicy,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Degrade gracefully when no target-availability-safe split exists.

    A long pseudo-out-of-sample run must not abort entirely because ONE early
    origin lacks enough target-available data to form a tuning split. We reuse
    the model's last successfully-tuned parameters when available, otherwise
    fall back to the model's registered defaults (empty override), and emit a
    RuntimeWarning naming the origin/horizon/model. A run-level flag records the
    degradation so ``run`` can still raise if NO origin ever tunes (a genuinely
    impossible configuration rather than a sparse early origin).
    """

    selection_cache[_SELECTION_DEGRADED_KEY] = True
    reused = cache_key in param_cache
    best_params = dict(param_cache.get(cache_key, {}))
    source = "last-tuned params" if reused else "model defaults"
    warnings.warn(
        f"model selection found no target-availability-safe validation split for "
        f"model {alias!r} at origin {origin_label} (horizon {target_step}); "
        f"falling back to {source} for this origin instead of tuning",
        RuntimeWarning,
        stacklevel=3,
    )
    cached_metadata = selection_cache.get(cache_key)
    base_metadata = (
        dict(cached_metadata) if isinstance(cached_metadata, dict) else {}
    )
    selection_metadata = {
        **base_metadata,
        "policy": selection_policy.to_dict(),
        "retuned": False,
        "selection_degraded": True,
        "selection_degraded_reason": "no_availability_safe_split",
        "selection_degraded_source": source,
    }
    return best_params, selection_metadata


def _assert_selection_was_possible(
    selection_cache: dict[str, Any],
    *,
    target_step: int,
) -> None:
    """Raise if model selection degraded at every origin (impossible config).

    Graceful degradation tolerates sparse early origins, but a configuration
    that can NEVER form a tuning split for ANY origin is a genuine
    misconfiguration that should still surface as an error.
    """

    ever_degraded = bool(selection_cache.get(_SELECTION_DEGRADED_KEY))
    ever_tuned = bool(selection_cache.get(_SELECTION_TUNED_KEY))
    if ever_degraded and not ever_tuned:
        raise ValueError(
            "model selection has no target-availability-safe validation splits "
            f"for horizon {target_step} at ANY origin; increase the available "
            "sample, move the validation window earlier, reduce the validation "
            "size, or reduce the forecast horizon"
        )


def _availability_safe_selection_splits(
    item: dict[str, Any],
    selection_index: pd.Index,
    *,
    target_step: int,
) -> list[Split]:
    if len(selection_index) == 0:
        return []
    window_spec = item.get("window_spec")
    if isinstance(window_spec, WindowSpec):
        val = window_spec.val
        base_embargo = (
            val.embargo
            if val.embargo is not None
            else window_spec.estimation.embargo
        )
        embargo = max(int(base_embargo), max(int(target_step) - 1, 0))
        min_train_size = (
            window_spec.min_train_size
            if window_spec.min_train_size is not None
            else val.min_train_size
        )
        try:
            return make_splitter(
                val.method,
                len(selection_index),
                validation_size=val.size,
                validation_ratio=val.ratio,
                min_train_size=min_train_size,
                n_splits=val.n_splits,
                step=val.step,
                horizon=val.horizon,
                random_state=val.random_state,
                embargo=embargo,
            )
        except ValueError:
            return _availability_safe_explicit_splits(
                item,
                selection_index,
                target_step=target_step,
            )
    return _availability_safe_explicit_splits(
        item,
        selection_index,
        target_step=target_step,
    )


def _allow_non_temporal_selection_splits(item: dict[str, Any]) -> bool:
    window_spec = item.get("window_spec")
    return isinstance(window_spec, WindowSpec) and window_spec.val.method == "random_kfold"


def _availability_safe_explicit_splits(
    item: dict[str, Any],
    selection_index: pd.Index,
    *,
    target_step: int,
) -> list[Split]:
    absolute_splits = item.get("absolute_val_splits") or []
    if not absolute_splits:
        return []
    base_index = _target_availability_base_index(item, selection_index)
    out: list[Split] = []
    for train_abs, val_abs in absolute_splits:
        train_abs_arr = np.asarray(train_abs, dtype=int)
        val_abs_arr = np.asarray(val_abs, dtype=int)
        val_abs_arr = val_abs_arr[
            (val_abs_arr >= 0)
            & (val_abs_arr < len(base_index))
        ]
        if len(val_abs_arr) == 0:
            continue
        val_labels = base_index[val_abs_arr]
        val_rel = selection_index.get_indexer(val_labels)
        val_keep = val_rel >= 0
        if not val_keep.any():
            continue
        kept_val_abs = val_abs_arr[val_keep]
        val_rel = val_rel[val_keep]
        train_cutoff = int(kept_val_abs.min()) - int(target_step)
        train_abs_arr = train_abs_arr[
            (train_abs_arr >= 0)
            & (train_abs_arr < len(base_index))
            & (train_abs_arr <= train_cutoff)
        ]
        if len(train_abs_arr) == 0:
            continue
        train_labels = base_index[train_abs_arr]
        train_rel = selection_index.get_indexer(train_labels)
        train_rel = train_rel[train_rel >= 0]
        if len(train_rel) == 0:
            continue
        out.append((train_rel.astype(int, copy=False), val_rel.astype(int, copy=False)))
    return out


def _align_feature_xy(X: Any, y: Any) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.DataFrame(X).copy()
    target = _single_target(y)
    temp_name = "__macroforecast_selection_target__"
    while temp_name in frame.columns:
        temp_name = f"_{temp_name}"
    aligned = pd.concat([target.rename(temp_name), frame], axis=1).dropna()
    aligned_target = aligned.pop(temp_name)
    aligned_target.name = target.name
    return aligned, aligned_target


def _relative_splits_for_index(
    splits: Sequence[Split],
    selection_index: pd.Index,
    base_index: pd.Index,
    *,
    allow_degenerate: bool = False,
) -> list[Split]:
    """Map absolute window positions onto a stage-local feature matrix.

    ``allow_degenerate`` controls what happens when feature alignment empties a
    fold's train or validation side. After data-dependent preprocessing (EM
    imputation that flags outliers as NaN) and lag-feature construction, some rows
    referenced by an absolute validation split may be dropped from the selection
    feature matrix, occasionally emptying one fold at a particular origin/horizon.
    With ``allow_degenerate=False`` (the default, used by the feature-set path) an
    empty fold is a hard error. With ``allow_degenerate=True`` (used by the
    per-origin selection-splits computation, which runs for every arm even those
    that never tune) the unusable fold is SKIPPED rather than aborting the whole
    multi-horizon run; ``select_params`` still raises downstream if an arm that
    needs tuning is left with no usable folds. Skipping an unusable CV fold is a
    standard, leak-free behaviour (it never feeds future rows into training).
    """

    if not splits:
        return []
    if not selection_index.is_unique:
        raise ValueError("selection feature index must be unique")
    out: list[Split] = []
    for split_id, (train_abs, val_abs) in enumerate(splits):
        train_labels = base_index[np.asarray(train_abs, dtype=int)]
        val_labels = base_index[np.asarray(val_abs, dtype=int)]
        train_pos = selection_index.get_indexer(train_labels)
        val_pos = selection_index.get_indexer(val_labels)
        train_pos = train_pos[train_pos >= 0]
        if len(train_pos) == 0:
            if allow_degenerate:
                continue
            raise ValueError(
                f"validation split {split_id} has no train rows after feature alignment"
            )
        val_pos = val_pos[val_pos >= 0]
        if len(val_pos) == 0:
            if allow_degenerate:
                continue
            raise ValueError(
                f"validation split {split_id} has no validation rows after feature alignment"
            )
        out.append((train_pos.astype(int, copy=False), val_pos.astype(int, copy=False)))
    return out


def _validate_selection_mapping(
    selection: SearchSpec | Mapping[str, SearchSpec | None] | None,
    model_runs: Sequence[_ModelRun],
) -> None:
    if selection is None or isinstance(selection, SearchSpec):
        return
    unknown = set(selection) - _run_keys(model_runs)
    if unknown:
        allowed = ", ".join(sorted(_run_keys(model_runs)))
        raise ValueError(
            f"model_selection contains keys that do not match a model alias or spec: "
            f"{sorted(unknown)}. Available keys: {allowed}."
        )


def _selection_for_model(
    selection: SearchSpec | Mapping[str, SearchSpec | None] | None,
    model_run: _ModelRun,
) -> tuple[SearchSpec | None, bool]:
    if selection is None or isinstance(selection, SearchSpec):
        return selection, True
    if model_run.alias in selection:
        return selection[model_run.alias], selection[model_run.alias] is not None
    if model_run.spec.name in selection:
        return selection[model_run.spec.name], selection[
            model_run.spec.name
        ] is not None
    return None, True
