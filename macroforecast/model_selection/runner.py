from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from macroforecast.model_selection.types import (
    ScoreAggregation,
    SearchTrial,
    _normalize_score_aggregation,
)
from macroforecast.window import Split

if TYPE_CHECKING:
    from macroforecast.models.specs import PrefixSearchSpec


def evaluate_candidate(
    model: Callable[..., Any],
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[Split],
    metric_fn: Callable[[Any, Any], float],
    fixed_params: dict[str, Any],
    params: dict[str, Any],
    trial: int,
    *,
    fold_ids: list[int] | tuple[int, ...] | None = None,
    score_aggregation: ScoreAggregation = "mean_split",
) -> SearchTrial:
    """Evaluate one parameter candidate across temporal validation splits."""

    trial_params = {**fixed_params, **params}
    aggregation = _normalize_score_aggregation(score_aggregation)
    resolved_fold_ids = _resolve_fold_ids(fold_ids, len(splits))
    scores: list[float] = []
    fold_truth: dict[int, list[pd.Series]] = {}
    fold_pred: dict[int, list[pd.Series]] = {}
    try:
        for split_id, (train_idx, val_idx) in enumerate(splits):
            fit = model(X.iloc[train_idx], y.iloc[train_idx], **trial_params)
            if not hasattr(fit, "predict"):
                raise TypeError("model callable must return an object with predict(X)")
            y_val = y.iloc[val_idx]
            pred = _prediction_series(fit.predict(X.iloc[val_idx]), index=y_val.index)
            if aggregation == "mean_split":
                scores.append(float(metric_fn(y_val, pred)))
            else:
                fold_id = resolved_fold_ids[split_id]
                fold_truth.setdefault(fold_id, []).append(y_val)
                fold_pred.setdefault(fold_id, []).append(pred)
        if aggregation == "mean_fold":
            for fold_id in dict.fromkeys(resolved_fold_ids):
                y_fold = pd.concat(fold_truth[fold_id])
                pred_fold = pd.concat(fold_pred[fold_id])
                scores.append(float(metric_fn(y_fold, pred_fold)))
    except Exception as exc:  # noqa: BLE001 - failed trials are part of search output.
        return SearchTrial(
            trial=trial,
            params=trial_params,
            score=np.nan,
            n_splits=len(splits),
            status="error",
            error=str(exc),
        )
    return SearchTrial(
        trial=trial,
        params=trial_params,
        score=float(np.mean(scores)),
        n_splits=len(splits),
        status="ok",
        error=None,
    )


def evaluate_candidate_group(
    prefix_search: "PrefixSearchSpec",
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[Split],
    metric_fn: Callable[[Any, Any], float],
    fixed_params: dict[str, Any],
    group_candidates: list[tuple[int, dict[str, Any]]],
    *,
    fold_ids: list[int] | tuple[int, ...] | None = None,
    score_aggregation: ScoreAggregation = "mean_split",
) -> list[SearchTrial]:
    """Evaluate a group of candidates that differ only in ``prefix_search.param``.

    Mirrors :func:`evaluate_candidate`'s control flow member-by-member, but shares
    one ``prefix_search.fit_prefix`` fit per validation split across the whole
    group (fit once at the group's max value of ``prefix_search.param``, read
    every member's prediction as a no-recompute prefix view). Output is bitwise
    identical to calling :func:`evaluate_candidate` once per group member.
    """

    aggregation = _normalize_score_aggregation(score_aggregation)
    resolved_fold_ids = _resolve_fold_ids(fold_ids, len(splits))

    trial_params: dict[int, dict[str, Any]] = {}
    k_by_trial: dict[int, int] = {}
    scores: dict[int, list[float]] = {}
    fold_truth: dict[int, dict[int, list[pd.Series]]] = {}
    fold_pred: dict[int, dict[int, list[pd.Series]]] = {}
    failed: dict[int, str | None] = {}
    for trial_id, candidate_params in group_candidates:
        params = {**fixed_params, **candidate_params}
        trial_params[trial_id] = params
        k_by_trial[trial_id] = int(params[prefix_search.param])
        scores[trial_id] = []
        fold_truth[trial_id] = {}
        fold_pred[trial_id] = {}
        failed[trial_id] = None

    first_trial_id = group_candidates[0][0]
    group_params = {
        key: value
        for key, value in trial_params[first_trial_id].items()
        if key != prefix_search.param
    }
    k_values = [k_by_trial[trial_id] for trial_id, _ in group_candidates]

    for split_id, (train_idx, val_idx) in enumerate(splits):
        if all(failed[trial_id] is not None for trial_id, _ in group_candidates):
            break
        try:
            views = prefix_search.fit_prefix(
                X.iloc[train_idx], y.iloc[train_idx], k_values, **group_params
            )
        except Exception as exc:  # noqa: BLE001 - shared-fit failure fails the group.
            message = str(exc)
            for trial_id, _ in group_candidates:
                if failed[trial_id] is None:
                    failed[trial_id] = message
                    scores[trial_id] = []
                    fold_truth[trial_id] = {}
                    fold_pred[trial_id] = {}
            continue
        for trial_id, _ in group_candidates:
            if failed[trial_id] is not None:
                continue
            try:
                y_val = y.iloc[val_idx]
                pred = _prediction_series(
                    views[k_by_trial[trial_id]].predict(X.iloc[val_idx]), index=y_val.index
                )
                if aggregation == "mean_split":
                    scores[trial_id].append(float(metric_fn(y_val, pred)))
                else:
                    fold_id = resolved_fold_ids[split_id]
                    fold_truth[trial_id].setdefault(fold_id, []).append(y_val)
                    fold_pred[trial_id].setdefault(fold_id, []).append(pred)
            except Exception as exc:  # noqa: BLE001 - per-candidate isolation.
                failed[trial_id] = str(exc)
                scores[trial_id] = []
                fold_truth[trial_id] = {}
                fold_pred[trial_id] = {}

    if aggregation == "mean_fold":
        for trial_id, _ in group_candidates:
            if failed[trial_id] is not None:
                continue
            try:
                for fold_id in dict.fromkeys(resolved_fold_ids):
                    y_fold = pd.concat(fold_truth[trial_id][fold_id])
                    pred_fold = pd.concat(fold_pred[trial_id][fold_id])
                    scores[trial_id].append(float(metric_fn(y_fold, pred_fold)))
            except Exception as exc:  # noqa: BLE001 - matches evaluate_candidate's
                # single try wrapping both the per-split loop and mean_fold aggregation.
                failed[trial_id] = str(exc)

    rows: list[SearchTrial] = []
    for trial_id, _ in group_candidates:
        if failed[trial_id] is not None:
            rows.append(
                SearchTrial(
                    trial=trial_id,
                    params=trial_params[trial_id],
                    score=float("nan"),
                    n_splits=len(splits),
                    status="error",
                    error=failed[trial_id],
                )
            )
        else:
            rows.append(
                SearchTrial(
                    trial=trial_id,
                    params=trial_params[trial_id],
                    score=float(np.mean(scores[trial_id])),
                    n_splits=len(splits),
                    status="ok",
                    error=None,
                )
            )
    return rows


def _prediction_series(value: Any, *, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        if len(value) != len(index):
            raise ValueError("prediction length must match validation rows")
        if value.index.equals(index):
            return value.astype(float).rename("prediction")
        return pd.Series(value.to_numpy(dtype=float), index=index, name="prediction")
    arr = np.asarray(value, dtype=float).reshape(-1)
    if len(arr) != len(index):
        raise ValueError("prediction length must match validation rows")
    return pd.Series(arr, index=index, name="prediction")


def _resolve_fold_ids(
    fold_ids: list[int] | tuple[int, ...] | None,
    n_splits: int,
) -> tuple[int, ...]:
    if fold_ids is None:
        return tuple(range(n_splits))
    resolved = tuple(int(fold_id) for fold_id in fold_ids)
    if len(resolved) != n_splits:
        raise ValueError("fold_ids length must match splits")
    return resolved


def trial_frame(rows: list[SearchTrial]) -> pd.DataFrame:
    """Return the public trial table shape from evaluated trial records."""

    frame = pd.DataFrame([row.to_record() for row in rows])
    if frame.empty:
        raise ValueError("parameter search produced no trials")
    first = ["trial"]
    last = ["score", "n_splits", "status", "error"]
    middle = [col for col in frame.columns if col not in set(first + last)]
    return frame[first + middle + last].sort_values("trial").reset_index(drop=True)


def parameter_columns(trials: pd.DataFrame) -> list[str]:
    """Return candidate parameter columns from a trial table."""

    reserved = {"trial", "score", "n_splits", "status", "error"}
    return [col for col in trials.columns if col not in reserved]


__all__ = [
    "evaluate_candidate",
    "evaluate_candidate_group",
    "parameter_columns",
    "trial_frame",
]
