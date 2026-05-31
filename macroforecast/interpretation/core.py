from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from importlib import import_module
from typing import Any

import numpy as np
import pandas as pd

from macroforecast.models import ModelFit

_INTERPRETATION_SCHEMA_VERSION = 1


def linear_coefficients(model: Any, *, sort: bool = True) -> pd.DataFrame:
    """Return native coefficients for linear-style fitted models."""

    fit = _coerce_fit(model)
    estimator = fit.estimator if isinstance(fit, ModelFit) else fit
    coef = getattr(estimator, "coef_", None)
    if coef is None:
        raise ValueError("model does not expose coef_")
    values = np.asarray(coef, dtype=float).reshape(-1)
    names = _feature_names(fit, len(values))
    table = pd.DataFrame(
        {
            "feature": names,
            "coefficient": values,
            "abs_coefficient": np.abs(values),
        }
    )
    if sort:
        table = table.sort_values("abs_coefficient", ascending=False, kind="stable")
    return _attach_schema(
        table.reset_index(drop=True),
        kind="linear_coefficients",
        model=model,
        method="native_coef",
        n_features=len(values),
    )


def tree_importance(model: Any, *, sort: bool = True) -> pd.DataFrame:
    """Return native tree importance for estimators exposing feature_importances_."""

    fit = _coerce_fit(model)
    estimator = fit.estimator if isinstance(fit, ModelFit) else fit
    importance = getattr(estimator, "feature_importances_", None)
    if importance is None:
        raise ValueError("model does not expose feature_importances_")
    values = np.asarray(importance, dtype=float).reshape(-1)
    names = _feature_names(fit, len(values))
    table = pd.DataFrame({"feature": names, "importance": values})
    if sort:
        table = table.sort_values("importance", ascending=False, kind="stable")
    return _attach_schema(
        table.reset_index(drop=True),
        kind="tree_importance",
        model=model,
        method="native_feature_importances",
        n_features=len(values),
    )


def permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    metric: Callable[[np.ndarray, np.ndarray], float] | str = "mse",
    n_repeats: int = 5,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Compute simple model-agnostic permutation importance.

    Importance is the degradation in the loss metric after permuting one
    feature. For score metrics where higher is better, pass a callable that
    already returns a loss-like value if positive degradation is desired.
    """

    if n_repeats <= 0:
        raise ValueError("n_repeats must be positive")
    frame = _as_feature_frame(X)
    target = np.asarray(y, dtype=float).reshape(-1)
    if len(frame) != len(target):
        raise ValueError("X and y must have the same number of rows")
    rng = np.random.default_rng(random_state)
    loss = _loss_func(metric)
    baseline = loss(target, _predict(model, frame))
    rows: list[dict[str, Any]] = []
    for feature in frame.columns:
        deltas = []
        for _ in range(int(n_repeats)):
            permuted = frame.copy()
            permuted[feature] = rng.permutation(permuted[feature].to_numpy())
            deltas.append(loss(target, _predict(model, permuted)) - baseline)
        values = np.asarray(deltas, dtype=float)
        rows.append(
            {
                "feature": str(feature),
                "importance": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "baseline_loss": float(baseline),
                "n_repeats": int(n_repeats),
            }
        )
    table = (
        pd.DataFrame(rows)
        .sort_values("importance", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    return _attach_schema(
        table,
        kind="permutation_importance",
        model=model,
        method="permutation_loss_degradation",
        n_features=frame.shape[1],
        metadata={
            "metric": getattr(loss, "__name__", str(metric)),
            "n_obs": int(len(frame)),
            "n_repeats": int(n_repeats),
        },
    )


def partial_dependence(
    model: Any,
    X: pd.DataFrame,
    *,
    features: Iterable[str] | str,
    grid_size: int = 20,
) -> pd.DataFrame:
    """Compute one-way manual partial-dependence curves."""

    frame = _as_feature_frame(X)
    selected = _resolve_features(frame, features)
    if grid_size <= 1:
        raise ValueError("grid_size must be greater than 1")
    rows: list[dict[str, Any]] = []
    for feature in selected:
        grid = np.linspace(
            float(frame[feature].min()),
            float(frame[feature].max()),
            int(grid_size),
        )
        for value in grid:
            replaced = frame.copy()
            replaced[feature] = value
            pred = _predict(model, replaced)
            rows.append(
                {
                    "feature": str(feature),
                    "value": float(value),
                    "prediction": float(np.mean(pred)),
                }
            )
    return _attach_schema(
        pd.DataFrame(rows),
        kind="partial_dependence",
        model=model,
        method="manual_one_way_pdp",
        n_features=len(selected),
        metadata={"grid_size": int(grid_size), "features": list(selected)},
    )


def accumulated_local_effect(
    model: Any,
    X: pd.DataFrame,
    *,
    feature: str,
    bins: int = 10,
) -> pd.DataFrame:
    """Compute a first-order accumulated local effect curve."""

    frame = _as_feature_frame(X)
    if feature not in frame.columns:
        raise ValueError(f"feature {feature!r} is not in X")
    if bins <= 1:
        raise ValueError("bins must be greater than 1")
    values = frame[feature].astype(float)
    edges = np.unique(np.quantile(values.dropna(), np.linspace(0.0, 1.0, int(bins) + 1)))
    if len(edges) < 3:
        raise ValueError("feature needs at least two non-empty ALE bins")
    effects = []
    centers = []
    for low, high in zip(edges[:-1], edges[1:], strict=False):
        mask = (values >= low) & (values <= high if high == edges[-1] else values < high)
        if not mask.any():
            effects.append(0.0)
            centers.append(float((low + high) / 2.0))
            continue
        lower = frame.loc[mask].copy()
        upper = lower.copy()
        lower[feature] = low
        upper[feature] = high
        effects.append(float(np.mean(_predict(model, upper) - _predict(model, lower))))
        centers.append(float((low + high) / 2.0))
    accumulated = np.cumsum(np.asarray(effects, dtype=float))
    accumulated = accumulated - accumulated.mean()
    table = pd.DataFrame(
        {
            "feature": str(feature),
            "bin": np.arange(1, len(accumulated) + 1),
            "center": centers,
            "ale": accumulated,
            "local_effect": effects,
        }
    )
    return _attach_schema(
        table,
        kind="accumulated_local_effect",
        model=model,
        method="first_order_ale",
        n_features=1,
        metadata={"feature": str(feature), "bins": int(bins)},
    )


def shap_values(
    model: Any,
    X: pd.DataFrame,
    *,
    background: pd.DataFrame | None = None,
    explainer: str = "auto",
    check_additivity: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Return SHAP values in a long pandas table.

    SHAP is an optional backend. Install ``macroforecast[interpretation]`` to
    use this helper.
    """

    shap = _optional_shap()
    frame = _as_feature_frame(X)
    background_frame = frame if background is None else _as_feature_frame(background)
    background_frame = background_frame.reindex(columns=frame.columns)
    resolved = _normalize_explainer(explainer)

    if resolved == "tree":
        target_model = model.estimator if isinstance(model, ModelFit) else model
        explainer_obj = shap.TreeExplainer(target_model, data=background_frame)
        explanation = explainer_obj.shap_values(frame, check_additivity=check_additivity)
        values = _coerce_shap_array(explanation, frame)
        base_values = _tree_base_values(explainer_obj, len(frame))
    else:
        predict_fn = lambda values: _predict(  # noqa: E731 - SHAP expects callable.
            model,
            _shap_prediction_frame(values, frame),
        )
        explainer_cls = (
            shap.PermutationExplainer if resolved == "permutation" else shap.Explainer
        )
        explainer_obj = explainer_cls(predict_fn, background_frame)
        call_kwargs = dict(kwargs)
        explanation = explainer_obj(frame, **call_kwargs)
        values = _coerce_shap_array(getattr(explanation, "values", explanation), frame)
        base_values = _coerce_base_values(
            getattr(explanation, "base_values", None),
            len(frame),
        )

    records: list[dict[str, Any]] = []
    for row_pos, (idx, row) in enumerate(frame.iterrows()):
        base_value = None if base_values is None else float(base_values[row_pos])
        for feature_pos, feature in enumerate(frame.columns):
            records.append(
                {
                    "row": int(row_pos),
                    "index": idx,
                    "feature": str(feature),
                    "feature_value": float(row.iloc[feature_pos]),
                    "shap_value": float(values[row_pos, feature_pos]),
                    "base_value": base_value,
                }
            )
    return _attach_schema(
        pd.DataFrame(records),
        kind="shap_values",
        model=model,
        method=f"shap_{resolved}",
        n_features=frame.shape[1],
        metadata={
            "explainer": resolved,
            "n_obs": int(len(frame)),
            "background_n_obs": int(len(background_frame)),
        },
    )


def custom_interpretation(
    model: Any,
    X: pd.DataFrame,
    func: Callable[..., Any],
    *,
    y: pd.Series | np.ndarray | None = None,
    name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    **params: Any,
) -> pd.DataFrame:
    """Run a user-supplied interpretation callable and attach metadata."""

    frame = _as_feature_frame(X)
    resolved_name = str(name or _callable_name(func) or "custom_interpretation")
    result = func(
        model,
        frame,
        y=y,
        metadata=dict(metadata or {}),
        **params,
    )
    table = _coerce_custom_table(result)
    return _attach_schema(
        table,
        kind="custom_interpretation",
        model=model,
        method=resolved_name,
        n_features=frame.shape[1],
        metadata={
            "name": resolved_name,
            "callable": _callable_name(func),
            "params": dict(params),
            "n_obs": int(len(frame)),
            "has_target": y is not None,
            "user_metadata": dict(metadata or {}),
        },
    )


def _coerce_fit(model: Any) -> Any:
    return model


def _feature_names(model: Any, n_features: int) -> list[str]:
    if isinstance(model, ModelFit) and model.feature_names:
        return list(model.feature_names)
    names = getattr(model, "feature_names_in_", None)
    if names is not None and len(names) == n_features:
        return [str(name) for name in names]
    return [f"x{i}" for i in range(n_features)]


def _as_feature_frame(X: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    return X.copy()


def _resolve_features(frame: pd.DataFrame, features: Iterable[str] | str) -> tuple[str, ...]:
    selected: tuple[str, ...]
    if isinstance(features, str):
        selected = (features,)
    else:
        selected = tuple(str(feature) for feature in features)
    missing = [feature for feature in selected if feature not in frame.columns]
    if missing:
        raise ValueError(f"features are not in X: {missing}")
    return selected


def _predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    if isinstance(model, ModelFit):
        return model.predict(X).to_numpy(dtype=float)
    if not hasattr(model, "predict"):
        raise ValueError("model must expose predict() or be a ModelFit")
    return np.asarray(model.predict(X), dtype=float).reshape(-1)


def _shap_prediction_frame(values: Any, template: pd.DataFrame) -> pd.DataFrame:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    index = template.index if arr.shape[0] == len(template) else None
    return pd.DataFrame(arr, columns=template.columns, index=index)


def _loss_func(metric: Callable[[np.ndarray, np.ndarray], float] | str) -> Callable[[np.ndarray, np.ndarray], float]:
    if callable(metric):
        return metric
    key = str(metric).lower()
    if key == "mse":
        return lambda y, pred: float(np.mean((y - pred) ** 2))
    if key == "mae":
        return lambda y, pred: float(np.mean(np.abs(y - pred)))
    raise ValueError("metric must be 'mse', 'mae', or a callable")


def _attach_schema(
    table: pd.DataFrame,
    *,
    kind: str,
    model: Any,
    method: str,
    n_features: int,
    metadata: dict[str, Any] | None = None,
) -> pd.DataFrame:
    table.attrs["macroforecast_metadata_schema"] = {
        "kind": kind,
        "version": _INTERPRETATION_SCHEMA_VERSION,
        "method": method,
        "model": _model_label(model),
        "n_features": int(n_features),
        "columns": [str(column) for column in table.columns],
        "metadata": dict(metadata or {}),
    }
    return table


def _coerce_custom_table(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, pd.Series):
        name = "value" if value.name is None else str(value.name)
        return value.rename(name).to_frame()
    if isinstance(value, Mapping):
        return pd.DataFrame([dict(value)])
    if isinstance(value, (list, tuple)):
        return pd.DataFrame(value)
    raise TypeError(
        "custom interpretation callable must return a DataFrame, Series, mapping, or sequence"
    )


def _callable_name(func: Any) -> str:
    return str(getattr(func, "__name__", func.__class__.__name__))


def _model_label(model: Any) -> str:
    if isinstance(model, ModelFit):
        return str(model.model)
    return f"{model.__class__.__module__}.{model.__class__.__qualname__}"


def _optional_shap() -> Any:
    try:
        return import_module("shap")
    except ImportError as exc:
        raise ImportError(
            "SHAP interpretation requires the optional shap backend. "
            "Install with `pip install 'macroforecast[interpretation]'`."
        ) from exc


def _normalize_explainer(explainer: str) -> str:
    key = str(explainer).lower().replace("-", "_")
    if key in {"auto", "permutation", "tree"}:
        return key
    raise ValueError("explainer must be 'auto', 'permutation', or 'tree'")


def _coerce_shap_array(values: Any, frame: pd.DataFrame) -> np.ndarray:
    if isinstance(values, list):
        if len(values) != 1:
            raise ValueError("multi-output SHAP values are not supported yet")
        values = values[0]
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    if arr.shape != frame.shape:
        raise ValueError(
            "SHAP output shape does not match X; expected "
            f"{frame.shape}, got {arr.shape}"
        )
    return arr


def _coerce_base_values(values: Any, n_obs: int) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return np.repeat(float(arr), n_obs)
    arr = arr.reshape(-1)
    if len(arr) == 1:
        return np.repeat(float(arr[0]), n_obs)
    if len(arr) != n_obs:
        return None
    return arr.astype(float, copy=False)


def _tree_base_values(explainer_obj: Any, n_obs: int) -> np.ndarray | None:
    expected = getattr(explainer_obj, "expected_value", None)
    if isinstance(expected, list):
        expected = expected[0] if len(expected) == 1 else None
    return _coerce_base_values(expected, n_obs)
