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
        frame = pd.DataFrame(
            arr, columns=[f"{name_prefix}{i}" for i in range(arr.shape[1])]
        )
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


def optional_import(
    module: str, *, extra: str | None = None, package: str | None = None
):
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
    collect_diagnostics: bool = True,
) -> ModelFit:
    frame, target = resolve_xy(X, y)
    estimator.fit(frame, target)
    diagnostics = (
        _fit_diagnostics(estimator, frame, target) if collect_diagnostics else {}
    )
    return ModelFit(
        estimator=estimator,
        model=model,
        feature_names=tuple(str(c) for c in frame.columns),
        target_name=str(target.name) if target.name is not None else None,
        metadata={"n_obs": len(frame), **(metadata or {})},
        diagnostics=diagnostics,
    )


def _fit_diagnostics(estimator: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """Collect model-specific diagnostics when the estimator exposes them."""

    diagnostics: dict[str, Any] = {}
    fitted = _safe_fitted_values(estimator, X, y.index)
    if fitted is not None:
        residuals = (y.astype(float) - fitted.astype(float)).rename("residual")
        diagnostics["fitted_values"] = fitted
        diagnostics["residuals"] = residuals
        diagnostics["metrics"] = _residual_metrics(residuals)

    coefficients = _coefficient_diagnostics(estimator, X.columns)
    if coefficients is not None:
        diagnostics["coefficients"] = coefficients
        selected = _nonzero_coefficient_features(coefficients)
        if selected:
            diagnostics.setdefault("selected_features", selected)

    intercept = getattr(estimator, "intercept_", None)
    if intercept is not None:
        diagnostics["intercept"] = _as_scalar_or_list(intercept)

    importance = _feature_importance_diagnostics(estimator, X.columns)
    if importance is not None:
        diagnostics["feature_importance"] = importance

    for attr in ("selected_features_", "factor_features_"):
        values = getattr(estimator, attr, None)
        if values is not None:
            diagnostics[attr.rstrip("_")] = tuple(str(value) for value in values)

    component_selected = getattr(estimator, "component_selected_features_", None)
    if component_selected is not None:
        diagnostics["component_selected_features"] = [
            tuple(str(value) for value in component) for component in component_selected
        ]

    loadings = _factor_loadings_diagnostics(estimator, X.columns)
    if loadings is not None:
        diagnostics["factor_loadings"] = loadings

    training_history = getattr(estimator, "training_history_", None)
    if training_history is not None:
        diagnostics["training_history"] = training_history

    sequence_context = getattr(estimator, "sequence_context_", None)
    if sequence_context:
        diagnostics["sequence_context"] = sequence_context

    return diagnostics


def _safe_fitted_values(
    estimator: Any, X: pd.DataFrame, index: pd.Index
) -> pd.Series | None:
    if not hasattr(estimator, "predict"):
        return None
    try:
        values = np.asarray(estimator.predict(X), dtype=float).reshape(-1)
    except Exception:  # noqa: BLE001 - diagnostics must never break model fitting.
        return None
    if len(values) != len(index):
        return None
    return pd.Series(values, index=index, name="fitted")


def _residual_metrics(residuals: pd.Series) -> dict[str, float | int]:
    values = residuals.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return {"n": 0}
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "mae": float(np.mean(np.abs(values))),
        "mse": float(np.mean(values**2)),
        "rmse": float(np.sqrt(np.mean(values**2))),
    }


def _coefficient_diagnostics(
    estimator: Any, columns: pd.Index
) -> pd.Series | pd.DataFrame | None:
    coef = getattr(estimator, "coef_", None)
    if coef is None:
        return None
    values = np.asarray(coef, dtype=float)
    if values.ndim == 0:
        return pd.Series([float(values)], index=["coef"], name="coefficient")
    if values.ndim == 1:
        index = (
            [str(column) for column in columns]
            if len(columns) == len(values)
            else [f"x{i}" for i in range(len(values))]
        )
        return pd.Series(values, index=index, name="coefficient")
    if values.ndim == 2:
        feature_index = (
            [str(column) for column in columns]
            if len(columns) == values.shape[1]
            else [f"x{i}" for i in range(values.shape[1])]
        )
        return pd.DataFrame(values, columns=feature_index)
    return None


def _nonzero_coefficient_features(
    coefficients: pd.Series | pd.DataFrame,
) -> tuple[str, ...]:
    if isinstance(coefficients, pd.DataFrame):
        mask = np.any(np.abs(coefficients.to_numpy(dtype=float)) > 1e-12, axis=0)
        return tuple(
            str(column) for column, keep in zip(coefficients.columns, mask) if keep
        )
    return tuple(
        str(index) for index, value in coefficients.items() if abs(float(value)) > 1e-12
    )


def _feature_importance_diagnostics(
    estimator: Any, columns: pd.Index
) -> pd.Series | None:
    values = getattr(estimator, "feature_importances_", None)
    if values is None:
        return None
    arr = np.asarray(values, dtype=float).reshape(-1)
    index = (
        [str(column) for column in columns]
        if len(columns) == len(arr)
        else [f"x{i}" for i in range(len(arr))]
    )
    return pd.Series(arr, index=index, name="feature_importance").sort_values(
        ascending=False
    )


def _factor_loadings_diagnostics(
    estimator: Any, columns: pd.Index
) -> pd.DataFrame | None:
    values = None
    for attr in ("factor_loadings_", "loadings_", "components_"):
        candidate = getattr(estimator, attr, None)
        if candidate is not None:
            values = np.asarray(candidate, dtype=float)
            break
    if values is None or values.ndim != 2:
        return None
    feature_names = tuple(getattr(estimator, "factor_features_", ())) or tuple(
        str(column) for column in columns
    )
    if len(feature_names) == values.shape[0]:
        index = list(feature_names)
        columns_out = [f"factor_{i + 1}" for i in range(values.shape[1])]
        return pd.DataFrame(values, index=index, columns=columns_out)
    if len(feature_names) == values.shape[1]:
        index = [f"factor_{i + 1}" for i in range(values.shape[0])]
        return pd.DataFrame(values, index=index, columns=list(feature_names))
    return pd.DataFrame(values)


def _as_scalar_or_list(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    return arr.tolist()


__all__ = [
    "align_xy",
    "as_frame",
    "as_series",
    "fit_estimator",
    "optional_import",
    "resolve_xy",
]
