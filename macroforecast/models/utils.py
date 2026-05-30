from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np
import pandas as pd

from macroforecast.feature_engineering import FeatureSet
from macroforecast.models.types import ModelFit


def as_frame(X: Any, *, name_prefix: str = "x") -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        frame = X.copy()
    else:
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError(f"X must be 1-D or 2-D, got shape {arr.shape!r}")
        frame = pd.DataFrame(arr, columns=[f"{name_prefix}{i}" for i in range(arr.shape[1])])
    if frame.shape[1] == 0:
        raise ValueError("X must contain at least one column")
    return frame


def as_series(y: Any, *, name: str = "y") -> pd.Series:
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("y DataFrame must contain exactly one column")
        series = y.iloc[:, 0].copy()
        series.name = series.name or name
        return series
    if isinstance(y, pd.Series):
        return y.copy()
    arr = np.asarray(y, dtype=float).reshape(-1)
    return pd.Series(arr, name=name)


def resolve_xy(X: Any, y: Any | None = None) -> tuple[pd.DataFrame, pd.Series]:
    if isinstance(X, FeatureSet):
        frame = X.X.copy()
        target_frame = X.y if y is None else y
        target = as_series(target_frame)
        return align_xy(frame, target)
    if y is None:
        raise TypeError("y is required unless X is a FeatureSet")
    return align_xy(as_frame(X), as_series(y))


def align_xy(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    target_name = y.name or "y"
    joined = pd.concat([X, y.rename("__target__")], axis=1).dropna()
    if joined.empty:
        raise ValueError("X and y have no aligned non-missing observations")
    frame = joined.drop(columns="__target__")
    target = joined["__target__"].rename(target_name)
    return frame, target


def optional_import(module: str, *, extra: str | None = None, package: str | None = None):
    try:
        return import_module(module)
    except ImportError as exc:
        install = package or module.split(".")[0]
        hint = f"pip install {install}"
        if extra:
            hint = f"pip install 'macroforecast[{extra}]'"
        raise ImportError(
            f"{module!r} is required for this model. Install with `{hint}`."
        ) from exc


def fit_estimator(
    estimator: Any,
    X: Any,
    y: Any | None,
    *,
    model: str,
    metadata: dict[str, Any] | None = None,
) -> ModelFit:
    frame, target = resolve_xy(X, y)
    estimator.fit(frame, target)
    return ModelFit(
        estimator=estimator,
        model=model,
        feature_names=tuple(str(c) for c in frame.columns),
        target_name=str(target.name) if target.name is not None else None,
        metadata={"n_obs": len(frame), **(metadata or {})},
    )


__all__ = [
    "align_xy",
    "as_frame",
    "as_series",
    "fit_estimator",
    "optional_import",
    "resolve_xy",
]
