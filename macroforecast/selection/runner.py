from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from macroforecast.selection.types import SearchTrial
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
) -> SearchTrial:
    """Evaluate one parameter candidate across temporal validation splits."""

    trial_params = {**fixed_params, **params}
    scores: list[float] = []
    try:
        for train_idx, val_idx in splits:
            fit = model(X.iloc[train_idx], y.iloc[train_idx], **trial_params)
            if not hasattr(fit, "predict"):
                raise TypeError("model callable must return an object with predict(X)")
            y_val = y.iloc[val_idx]
            pred = _prediction_series(fit.predict(X.iloc[val_idx]), index=y_val.index)
            scores.append(float(metric_fn(y_val, pred)))
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
        n_splits=len(scores),
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
