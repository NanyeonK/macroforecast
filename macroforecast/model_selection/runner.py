from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from macroforecast.model_selection.types import (
    ScoreAggregation,
    SearchTrial,
    _normalize_score_aggregation,
)
from macroforecast.window import Split


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


__all__ = ["evaluate_candidate", "parameter_columns", "trial_frame"]
