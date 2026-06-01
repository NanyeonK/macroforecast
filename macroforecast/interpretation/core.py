from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import combinations
from importlib import import_module
from math import comb
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


def model_native_linear_coef(model: Any, *, sort: bool = True) -> pd.DataFrame:
    """Alias for legacy L7 naming: native linear coefficients."""

    table = linear_coefficients(model, sort=sort)
    table.attrs["macroforecast_metadata_schema"]["method"] = "model_native_linear_coef"
    return table


def model_native_tree_importance(model: Any, *, sort: bool = True) -> pd.DataFrame:
    """Alias for legacy L7 naming: native tree feature importance."""

    table = tree_importance(model, sort=sort)
    table.attrs["macroforecast_metadata_schema"]["method"] = "model_native_tree_importance"
    return table


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


def permutation_importance_strobl(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    metric: Callable[[np.ndarray, np.ndarray], float] | str = "mse",
    n_repeats: int = 5,
    n_bins: int = 5,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Conditional permutation importance following the Strobl idea.

    Each feature is permuted within bins of its most correlated companion
    feature. This keeps the permutation closer to the observed conditional
    distribution than a marginal shuffle, which is the relevant distinction
    when macro predictors are strongly collinear.
    """

    if n_repeats <= 0:
        raise ValueError("n_repeats must be positive")
    if n_bins <= 1:
        raise ValueError("n_bins must be greater than 1")
    frame = _as_feature_frame(X)
    target = pd.Series(np.asarray(y, dtype=float).reshape(-1), index=frame.index)
    if len(frame) != len(target):
        raise ValueError("X and y must have the same number of rows")
    joined = pd.concat([frame, target.rename("__target__")], axis=1).dropna()
    eval_x = joined.loc[:, frame.columns]
    eval_y = joined["__target__"].to_numpy(dtype=float)
    rng = np.random.default_rng(random_state)
    loss = _loss_func(metric)
    baseline = loss(eval_y, _predict(model, eval_x))
    rows: list[dict[str, Any]] = []
    for feature in eval_x.columns:
        other = [column for column in eval_x.columns if column != feature]
        conditioning_feature = None
        if other:
            correlations = eval_x[other].corrwith(eval_x[feature]).abs().fillna(0.0)
            conditioning_feature = str(correlations.idxmax())
        deltas: list[float] = []
        for _ in range(int(n_repeats)):
            permuted = eval_x.copy()
            if conditioning_feature is None:
                permuted[feature] = rng.permutation(permuted[feature].to_numpy())
            else:
                bins = _safe_qcut(eval_x[conditioning_feature], int(n_bins))
                for bin_id in bins.dropna().unique():
                    mask = bins == bin_id
                    if int(mask.sum()) <= 1:
                        continue
                    values = permuted.loc[mask, feature].to_numpy()
                    permuted.loc[mask, feature] = rng.permutation(values)
            deltas.append(loss(eval_y, _predict(model, permuted)) - baseline)
        values = np.asarray(deltas, dtype=float)
        rows.append(
            {
                "feature": str(feature),
                "importance": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "baseline_loss": float(baseline),
                "n_repeats": int(n_repeats),
                "conditioning_feature": conditioning_feature,
                "n_bins": int(n_bins),
            }
        )
    table = (
        pd.DataFrame(rows)
        .sort_values("importance", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    return _attach_schema(
        table,
        kind="permutation_importance_strobl",
        model=model,
        method="conditional_permutation_loss_degradation",
        n_features=frame.shape[1],
        metadata={
            "metric": getattr(loss, "__name__", str(metric)),
            "n_obs": int(len(eval_x)),
            "n_repeats": int(n_repeats),
            "n_bins": int(n_bins),
        },
    )


def lofo_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    fit_func: Callable[[pd.DataFrame, pd.Series], Any] | None = None,
    metric: Callable[[np.ndarray, np.ndarray], float] | str = "mse",
    sort: bool = True,
) -> pd.DataFrame:
    """Leave-one-feature-out importance.

    If ``fit_func`` is supplied, the model is refit without each feature. If it
    is omitted, the already fitted model is evaluated after setting the held-out
    feature to zero. The latter is a prediction-drop diagnostic, not a refit
    LOFO experiment, and the returned metadata records that mode explicitly.
    """

    frame = _as_feature_frame(X)
    target = pd.Series(np.asarray(y, dtype=float).reshape(-1), index=frame.index)
    if len(frame) != len(target):
        raise ValueError("X and y must have the same number of rows")
    loss = _loss_func(metric)
    baseline_model = fit_func(frame, target) if fit_func is not None else model
    baseline = loss(target.to_numpy(dtype=float), _predict(baseline_model, frame))
    rows: list[dict[str, Any]] = []
    mode = "refit" if fit_func is not None else "prediction_drop"
    for feature in frame.columns:
        if fit_func is not None:
            reduced_x = frame.drop(columns=[feature])
            reduced_model = fit_func(reduced_x, target)
            pred = _predict(reduced_model, reduced_x)
        else:
            reduced_x = frame.copy()
            reduced_x[feature] = 0.0
            pred = _predict(model, reduced_x)
        heldout_loss = loss(target.to_numpy(dtype=float), pred)
        rows.append(
            {
                "feature": str(feature),
                "importance": float(heldout_loss - baseline),
                "baseline_loss": float(baseline),
                "heldout_loss": float(heldout_loss),
                "mode": mode,
            }
        )
    table = pd.DataFrame(rows)
    if sort:
        table = table.sort_values("importance", ascending=False, kind="stable")
    return _attach_schema(
        table.reset_index(drop=True),
        kind="lofo_importance",
        model=baseline_model,
        method=mode,
        n_features=frame.shape[1],
        metadata={"metric": getattr(loss, "__name__", str(metric)), "mode": mode},
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


def friedman_h_interaction(
    model: Any,
    X: pd.DataFrame,
    *,
    features: Sequence[str] | None = None,
    grid_size: int = 10,
) -> pd.DataFrame:
    """Compute pairwise Friedman-Popescu H interaction statistics.

    The implementation uses manual one-way and two-way partial dependence on a
    regular grid. Values are bounded to ``[0, inf)`` by construction; larger
    values indicate stronger interaction relative to the pair's joint partial
    dependence variation.
    """

    frame = _as_feature_frame(X)
    selected = tuple(features) if features is not None else tuple(str(c) for c in frame.columns)
    _resolve_features(frame, selected)
    if grid_size <= 1:
        raise ValueError("grid_size must be greater than 1")
    rows: list[dict[str, Any]] = []
    for left, right in combinations(selected, 2):
        left_grid = _grid_values(frame[left], int(grid_size))
        right_grid = _grid_values(frame[right], int(grid_size))
        joint = np.empty((len(left_grid), len(right_grid)), dtype=float)
        for i, left_value in enumerate(left_grid):
            for j, right_value in enumerate(right_grid):
                replaced = frame.copy()
                replaced[left] = left_value
                replaced[right] = right_value
                joint[i, j] = float(np.mean(_predict(model, replaced)))
        left_pd = joint.mean(axis=1, keepdims=True)
        right_pd = joint.mean(axis=0, keepdims=True)
        centered = joint - left_pd - right_pd + joint.mean()
        denom = float(np.var(joint))
        h_value = 0.0 if denom <= 1e-15 else float(np.sqrt(max(np.var(centered) / denom, 0.0)))
        rows.append(
            {
                "feature_1": str(left),
                "feature_2": str(right),
                "h_statistic": h_value,
                "joint_variance": denom,
                "interaction_variance": float(np.var(centered)),
                "grid_size": int(grid_size),
            }
        )
    return _attach_schema(
        pd.DataFrame(rows).sort_values("h_statistic", ascending=False, kind="stable").reset_index(drop=True),
        kind="friedman_h_interaction",
        model=model,
        method="manual_partial_dependence_h",
        n_features=len(selected),
        metadata={"features": list(selected), "grid_size": int(grid_size)},
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


def shap_importance(
    model: Any,
    X: pd.DataFrame,
    *,
    background: pd.DataFrame | None = None,
    explainer: str = "auto",
    check_additivity: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Summarize SHAP values as global mean absolute feature importance."""

    values = shap_values(
        model,
        X,
        background=background,
        explainer=explainer,
        check_additivity=check_additivity,
        **kwargs,
    )
    grouped = values.groupby("feature", as_index=False).agg(
        importance=("shap_value", lambda item: float(np.mean(np.abs(item)))),
        mean_shap=("shap_value", "mean"),
        std_shap=("shap_value", "std"),
    )
    grouped["std_shap"] = grouped["std_shap"].fillna(0.0)
    grouped = grouped.sort_values("importance", ascending=False, kind="stable").reset_index(drop=True)
    return _attach_schema(
        grouped,
        kind="shap_importance",
        model=model,
        method=f"shap_{_normalize_explainer(explainer)}_global_importance",
        n_features=_as_feature_frame(X).shape[1],
        metadata=values.attrs.get("macroforecast_metadata_schema", {}).get("metadata", {}),
    )


def shap_tree(model: Any, X: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    """Tree SHAP global importance using the optional ``shap`` backend."""

    return shap_importance(model, X, explainer="tree", **kwargs)


def shap_linear(model: Any, X: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    """Linear SHAP-style global importance using ``shap.Explainer``."""

    return shap_importance(model, X, explainer="auto", **kwargs)


def shap_kernel(model: Any, X: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    """Kernel/permutation SHAP-style global importance."""

    return shap_importance(model, X, explainer="permutation", **kwargs)


def shap_deep(model: Any, X: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    """Deep-model SHAP-style global importance.

    This callable uses the generic SHAP explainer path because deep backends
    vary by installed torch/shap version. Gradient-specific methods are exposed
    separately and require ``captum``.
    """

    return shap_importance(model, X, explainer="auto", **kwargs)


def forecast_decomposition(
    model: Any,
    X: pd.DataFrame,
    *,
    row: int | str | pd.Timestamp = -1,
    sort: bool = True,
) -> pd.DataFrame:
    """Decompose one prediction into linear feature contributions."""

    frame = _as_feature_frame(X)
    selected = _select_row(frame, row)
    estimator = _estimator(model)
    coef = getattr(estimator, "coef_", None)
    if coef is None:
        fallback = tree_importance(model, sort=sort).rename(columns={"importance": "abs_contribution"})
        fallback["contribution"] = fallback["abs_contribution"]
        fallback["feature_value"] = np.nan
        fallback["coefficient"] = np.nan
        fallback.attrs["macroforecast_metadata_schema"]["kind"] = "forecast_decomposition"
        fallback.attrs["macroforecast_metadata_schema"]["method"] = "tree_importance_fallback"
        return fallback
    values = np.asarray(coef, dtype=float).reshape(-1)
    names = _feature_names(model, len(values))
    selected = selected.reindex(names, fill_value=0.0).astype(float)
    contribution = selected.to_numpy(dtype=float) * values
    table = pd.DataFrame(
        {
            "feature": names,
            "feature_value": selected.to_numpy(dtype=float),
            "coefficient": values,
            "contribution": contribution,
            "abs_contribution": np.abs(contribution),
        }
    )
    intercept = getattr(estimator, "intercept_", None)
    if intercept is not None:
        table = pd.concat(
            [
                table,
                pd.DataFrame(
                    [
                        {
                            "feature": "__intercept__",
                            "feature_value": 1.0,
                            "coefficient": float(np.asarray(intercept).reshape(-1)[0]),
                            "contribution": float(np.asarray(intercept).reshape(-1)[0]),
                            "abs_contribution": abs(float(np.asarray(intercept).reshape(-1)[0])),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    if sort:
        table = table.sort_values("abs_contribution", ascending=False, kind="stable")
    return _attach_schema(
        table.reset_index(drop=True),
        kind="forecast_decomposition",
        model=model,
        method="linear_contribution",
        n_features=len(names),
        metadata={"row": _jsonish_index(selected.name)},
    )


def cumulative_r2_contribution(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    feature_order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Sequential contribution of features to in-sample prediction R-squared."""

    frame = _as_feature_frame(X)
    target = np.asarray(y, dtype=float).reshape(-1)
    if len(frame) != len(target):
        raise ValueError("X and y must have the same number of rows")
    if feature_order is None:
        try:
            order = list(linear_coefficients(model)["feature"])
        except Exception:
            order = list(permutation_importance(model, frame, target, n_repeats=1)["feature"])
    else:
        order = list(_resolve_features(frame, feature_order))
    active = pd.DataFrame(0.0, index=frame.index, columns=frame.columns)
    previous = _r2_score(target, np.repeat(float(np.mean(target)), len(target)))
    rows: list[dict[str, Any]] = []
    for step, feature in enumerate(order, start=1):
        active[feature] = frame[feature]
        current = _r2_score(target, _predict(model, active))
        rows.append(
            {
                "step": int(step),
                "feature": str(feature),
                "r2": float(current),
                "incremental_r2": float(current - previous),
                "cumulative_features": int(step),
            }
        )
        previous = current
    return _attach_schema(
        pd.DataFrame(rows),
        kind="cumulative_r2_contribution",
        model=model,
        method="sequential_zero_fill_prediction",
        n_features=frame.shape[1],
        metadata={"feature_order": order},
    )


def group_aggregate(
    table: pd.DataFrame,
    *,
    groups: Mapping[str, str | Sequence[str]] | None = None,
    group_column: str | None = None,
    value_column: str | None = None,
    aggregation: str = "sum",
) -> pd.DataFrame:
    """Aggregate feature-level importance into user or metadata groups."""

    frame = table.copy()
    if "feature" not in frame.columns:
        raise ValueError("table must contain a 'feature' column")
    value = value_column or _infer_importance_column(frame)
    if group_column is not None:
        if group_column not in frame.columns:
            raise ValueError(f"group_column {group_column!r} is not in table")
        frame["group"] = frame[group_column].astype(str)
    else:
        mapping = _normalize_group_mapping(groups)
        frame["group"] = frame["feature"].map(lambda item: mapping.get(str(item), str(item).split("_")[0]))
    grouped = _aggregate_importance(frame, group_by="group", value_column=value, aggregation=aggregation)
    return _attach_schema(
        grouped,
        kind="group_aggregate",
        model=None,
        method=aggregation,
        n_features=int(frame["feature"].nunique()),
        metadata={"value_column": value, "aggregation": aggregation},
    )


def lineage_attribution(
    table: pd.DataFrame,
    lineage: Mapping[str, Any],
    *,
    level: str = "pipeline_name",
    value_column: str | None = None,
    aggregation: str = "sum",
) -> pd.DataFrame:
    """Aggregate feature importance using feature-lineage metadata."""

    frame = table.copy()
    if "feature" not in frame.columns:
        raise ValueError("table must contain a 'feature' column")
    value = value_column or _infer_importance_column(frame)

    def resolve(feature: Any) -> str:
        meta = lineage.get(str(feature), {})
        if isinstance(meta, Mapping):
            return str(meta.get(level, meta.get("pipeline", meta.get("source", "unknown"))))
        return str(meta or "unknown")

    frame["lineage"] = frame["feature"].map(resolve)
    grouped = _aggregate_importance(frame, group_by="lineage", value_column=value, aggregation=aggregation)
    grouped = grouped.rename(columns={"lineage": level})
    return _attach_schema(
        grouped,
        kind="lineage_attribution",
        model=None,
        method=aggregation,
        n_features=int(frame["feature"].nunique()),
        metadata={"level": level, "value_column": value, "aggregation": aggregation},
    )


def transformation_attribution(
    evaluation: pd.DataFrame,
    *,
    pipeline_column: str | None = None,
    metric: str | None = None,
    method: str = "shapley_over_pipelines",
    target_columns: Sequence[str] = ("target", "horizon"),
) -> pd.DataFrame:
    """Attribute forecast loss differences to preprocessing/feature pipelines."""

    frame = evaluation.copy()
    pipeline_col = pipeline_column or _first_present(frame, ("pipeline", "pipeline_id", "model", "model_id"))
    if pipeline_col is None:
        raise ValueError("evaluation must contain a pipeline/model column")
    metric_col = metric or _first_present(frame, ("mse", "rmse", "mae", "loss", "score"))
    if metric_col is None:
        raise ValueError("evaluation must contain a metric column")
    if method not in {"shapley_over_pipelines", "marginal_addition", "leave_one_out_pipeline"}:
        raise ValueError("method must be 'shapley_over_pipelines', 'marginal_addition', or 'leave_one_out_pipeline'")
    group_cols = [column for column in target_columns if column in frame.columns]
    grouped_iter = frame.groupby(group_cols, dropna=False) if group_cols else [((), frame)]
    rows: list[dict[str, Any]] = []
    for key, group in grouped_iter:
        losses = group.groupby(pipeline_col, as_index=True)[metric_col].mean().astype(float)
        pipelines = list(losses.index.astype(str))
        values = losses.to_numpy(dtype=float)
        if len(pipelines) == 0:
            continue
        contribution = _pipeline_loss_contribution(values, method=method)
        base: dict[str, Any] = {}
        if group_cols:
            key_tuple = key if isinstance(key, tuple) else (key,)
            base = dict(zip(group_cols, key_tuple, strict=False))
        for pipeline, loss_value, contrib in zip(pipelines, values, contribution, strict=False):
            rows.append(
                {
                    **base,
                    "pipeline": pipeline,
                    "loss": float(loss_value),
                    "contribution": float(contrib),
                    "method": method,
                    "metric": metric_col,
                }
            )
    return _attach_schema(
        pd.DataFrame(rows),
        kind="transformation_attribution",
        model=None,
        method=method,
        n_features=0,
        metadata={"pipeline_column": pipeline_col, "metric": metric_col, "group_columns": group_cols},
    )


def attention_weights(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | None = None,
    *,
    add_intercept: bool = True,
    ridge: float = 1e-8,
) -> pd.DataFrame:
    """OLS attention weights ``Omega = X_test (X_train'X_train)^-1 X_train'``."""

    train = _as_feature_frame(X_train).astype(float)
    test = train if X_test is None else _as_feature_frame(X_test).reindex(columns=train.columns, fill_value=0.0).astype(float)
    train_matrix = _design_matrix(train, add_intercept=add_intercept)
    test_matrix = _design_matrix(test, add_intercept=add_intercept)
    gram = train_matrix.T @ train_matrix
    if ridge > 0:
        gram = gram + float(ridge) * np.eye(gram.shape[0])
    omega = test_matrix @ np.linalg.pinv(gram) @ train_matrix.T
    rows: list[dict[str, Any]] = []
    for test_pos, test_index in enumerate(test.index):
        for train_pos, train_index in enumerate(train.index):
            rows.append(
                {
                    "test_row": int(test_pos),
                    "test_index": test_index,
                    "train_row": int(train_pos),
                    "train_index": train_index,
                    "weight": float(omega[test_pos, train_pos]),
                }
            )
    table = pd.DataFrame(rows)
    table.attrs["attention_matrix"] = omega
    return _attach_schema(
        table,
        kind="attention_weights",
        model=None,
        method="ols_closed_form",
        n_features=train.shape[1],
        metadata={
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "add_intercept": bool(add_intercept),
            "ridge": float(ridge),
        },
    )


def dual_decomposition(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_test: pd.DataFrame | None = None,
    *,
    add_intercept: bool = True,
    ridge: float = 1e-8,
) -> pd.DataFrame:
    """Represent OLS predictions as weighted sums of training outcomes."""

    train = _as_feature_frame(X_train)
    target = np.asarray(y_train, dtype=float).reshape(-1)
    if len(train) != len(target):
        raise ValueError("X_train and y_train must have the same number of rows")
    weights = attention_weights(train, X_test, add_intercept=add_intercept, ridge=ridge)
    table = weights.copy()
    table["train_y"] = table["train_row"].map(lambda row: float(target[int(row)]))
    table["contribution"] = table["weight"] * table["train_y"]
    summary = (
        table.groupby(["test_row", "test_index"], as_index=False)
        .agg(prediction=("contribution", "sum"), gross_weight=("weight", lambda item: float(np.abs(item).sum())))
    )
    table.attrs["prediction_summary"] = summary
    return _attach_schema(
        table,
        kind="dual_decomposition",
        model=None,
        method="ols_dual_attention",
        n_features=train.shape[1],
        metadata={"add_intercept": bool(add_intercept), "ridge": float(ridge)},
    )


def generalized_irf(model: Any, *, n_periods: int = 12, target: str | int | None = None) -> pd.DataFrame:
    """Pesaran-Shin generalized impulse response importance for VAR models."""

    return _var_irf_table(model, n_periods=n_periods, target=target, method="generalized_irf")


def orthogonalised_irf(model: Any, *, n_periods: int = 12, target: str | int | None = None) -> pd.DataFrame:
    """Cholesky orthogonalised impulse response importance for VAR models."""

    return _var_irf_table(model, n_periods=n_periods, target=target, method="orthogonalised_irf")


def fevd(model: Any, *, n_periods: int = 12, target: str | int | None = None) -> pd.DataFrame:
    """Forecast error variance decomposition importance for VAR models."""

    results = _var_results(model)
    names = _var_names(results)
    target_pos = _target_position(names, target)
    try:
        decomp = np.asarray(results.fevd(int(n_periods)).decomp, dtype=float)
        # statsmodels shape is typically (equation, horizon, shock).
        values = decomp[target_pos, : int(n_periods), :].sum(axis=0)
    except Exception:
        return orthogonalised_irf(model, n_periods=n_periods, target=target)
    table = pd.DataFrame(
        {
            "feature": names[: len(values)],
            "importance": [float(abs(v)) for v in values],
            "coefficient": [None] * len(values),
            "status": "operational",
        }
    )
    return _attach_schema(
        table.sort_values("importance", ascending=False, kind="stable").reset_index(drop=True),
        kind="fevd",
        model=model,
        method="statsmodels_fevd",
        n_features=len(values),
        metadata={"n_periods": int(n_periods), "target": names[target_pos]},
    )


def historical_decomposition(
    model: Any,
    *,
    max_lag: int = 12,
    target: str | int | None = None,
) -> pd.DataFrame:
    """Reduced-form VAR historical contribution summary."""

    results = _var_results(model)
    names = _var_names(results)
    target_pos = _target_position(names, target)
    ma = np.asarray(results.ma_rep(maxn=int(max_lag)), dtype=float)
    resid = np.asarray(results.resid, dtype=float)
    rows: list[dict[str, Any]] = []
    for shock_pos, name in enumerate(names):
        path = np.zeros(resid.shape[0], dtype=float)
        for t in range(resid.shape[0]):
            total = 0.0
            for lag in range(min(int(max_lag), t) + 1):
                total += float(ma[lag, target_pos, shock_pos] * resid[t - lag, shock_pos])
            path[t] = total
        rows.append(
            {
                "feature": str(name),
                "importance": float(np.mean(np.abs(path))),
                "mean_contribution": float(np.mean(path)),
                "max_abs_contribution": float(np.max(np.abs(path))) if len(path) else 0.0,
            }
        )
    return _attach_schema(
        pd.DataFrame(rows).sort_values("importance", ascending=False, kind="stable").reset_index(drop=True),
        kind="historical_decomposition",
        model=model,
        method="reduced_form_ma_residual_contribution",
        n_features=len(names),
        metadata={"max_lag": int(max_lag), "target": names[target_pos]},
    )


def lasso_inclusion_frequency(
    model: Any,
    X: pd.DataFrame | None = None,
    y: pd.Series | np.ndarray | None = None,
    *,
    fit_func: Callable[[pd.DataFrame, pd.Series], Any] | None = None,
    n_bootstraps: int = 50,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Estimate coefficient nonzero frequency for lasso-style models."""

    if X is None or y is None or fit_func is None:
        coef = linear_coefficients(model, sort=False)
        coef["inclusion_frequency"] = (coef["coefficient"].abs() > 1e-9).astype(float)
        coef["importance"] = coef["inclusion_frequency"]
        coef.attrs["macroforecast_metadata_schema"]["kind"] = "lasso_inclusion_frequency"
        coef.attrs["macroforecast_metadata_schema"]["method"] = "single_fit_nonzero"
        return coef.sort_values("importance", ascending=False, kind="stable").reset_index(drop=True)
    frame = _as_feature_frame(X)
    target = pd.Series(np.asarray(y, dtype=float).reshape(-1), index=frame.index)
    if len(frame) != len(target):
        raise ValueError("X and y must have the same number of rows")
    if n_bootstraps <= 0:
        raise ValueError("n_bootstraps must be positive")
    rng = np.random.default_rng(random_state)
    counts = pd.Series(0.0, index=frame.columns)
    for _ in range(int(n_bootstraps)):
        sample_pos = rng.integers(0, len(frame), size=len(frame))
        sample_x = frame.iloc[sample_pos]
        sample_y = target.iloc[sample_pos]
        fit = fit_func(sample_x, sample_y)
        coef = linear_coefficients(fit, sort=False).set_index("feature")["coefficient"]
        counts = counts.add((coef.abs() > 1e-9).astype(float), fill_value=0.0)
    table = pd.DataFrame(
        {
            "feature": counts.index.astype(str),
            "inclusion_frequency": (counts / float(n_bootstraps)).to_numpy(dtype=float),
        }
    )
    table["importance"] = table["inclusion_frequency"]
    return _attach_schema(
        table.sort_values("importance", ascending=False, kind="stable").reset_index(drop=True),
        kind="lasso_inclusion_frequency",
        model=model,
        method="bootstrap_nonzero_frequency",
        n_features=frame.shape[1],
        metadata={"n_bootstraps": int(n_bootstraps)},
    )


def mrf_gtvp(model: Any, X: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return Macroeconomic Random Forest GTVP coefficient paths.

    The vendored MacroRandomForest backend emits ``betas`` after prediction.
    This callable exposes those paths directly instead of reducing them to a
    static forest importance score.
    """

    estimator = _estimator(model)
    if getattr(estimator, "output_", None) is None and X is not None:
        _predict(model, _as_feature_frame(X))
    output = getattr(estimator, "output_", None)
    if not isinstance(output, Mapping):
        raise ValueError("mrf_gtvp requires a macro_random_forest fit after predict()")
    betas = output.get("betas")
    if betas is None:
        raise ValueError("macro_random_forest output does not contain 'betas'")
    arr = np.asarray(betas, dtype=float)
    if arr.ndim != 2:
        raise ValueError("macro_random_forest 'betas' must be a 2-D array")
    names = _mrf_beta_names(estimator, output, arr.shape[1])
    if X is not None and len(X) == arr.shape[0]:
        index = list(_as_feature_frame(X).index)
    else:
        index = list(range(arr.shape[0]))
    rows: list[dict[str, Any]] = []
    for row_pos, idx in enumerate(index):
        for col_pos, name in enumerate(names):
            coef = float(arr[row_pos, col_pos])
            rows.append(
                {
                    "row": int(row_pos),
                    "index": idx,
                    "feature": str(name),
                    "coefficient": coef,
                    "abs_coefficient": abs(coef),
                    "importance": abs(coef),
                }
            )
    table = pd.DataFrame(rows)
    summary = (
        table.groupby("feature", as_index=False)
        .agg(
            importance=("abs_coefficient", "mean"),
            mean_coefficient=("coefficient", "mean"),
            std_coefficient=("coefficient", "std"),
        )
        .fillna({"std_coefficient": 0.0})
        .sort_values("importance", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    table.attrs["summary"] = summary
    return _attach_schema(
        table,
        kind="mrf_gtvp",
        model=model,
        method="macro_random_forest_beta_path",
        n_features=max(0, len(names) - 1),
        metadata={"n_rows": int(arr.shape[0]), "beta_columns": names},
    )


def rolling_recompute(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    window: int | None = None,
    step: int | None = None,
    method: str = "permutation_importance",
    n_repeats: int = 1,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Recompute feature importance on rolling evaluation windows."""

    frame = _as_feature_frame(X)
    target = pd.Series(np.asarray(y, dtype=float).reshape(-1), index=frame.index)
    if len(frame) != len(target):
        raise ValueError("X and y must have the same number of rows")
    if method not in {"permutation_importance", "permutation_importance_strobl"}:
        raise ValueError("method must be 'permutation_importance' or 'permutation_importance_strobl'")
    width = int(window or max(8, len(frame) // 4))
    if width <= 1:
        raise ValueError("window must be greater than 1")
    stride = int(step or max(1, width // 4))
    if stride <= 0:
        raise ValueError("step must be positive")
    rows: list[pd.DataFrame] = []
    for end in range(width, len(frame) + 1, stride):
        sub_x = frame.iloc[end - width : end]
        sub_y = target.iloc[end - width : end]
        if method == "permutation_importance_strobl":
            table = permutation_importance_strobl(
                model,
                sub_x,
                sub_y,
                n_repeats=n_repeats,
                random_state=random_state,
            )
        else:
            table = permutation_importance(
                model,
                sub_x,
                sub_y,
                n_repeats=n_repeats,
                random_state=random_state,
            )
        table = table.copy()
        table.insert(0, "window_end", sub_x.index[-1])
        table.insert(0, "window_start", sub_x.index[0])
        table.insert(0, "window_id", len(rows))
        rows.append(table)
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return _attach_schema(
        result,
        kind="rolling_recompute",
        model=model,
        method=method,
        n_features=frame.shape[1],
        metadata={"window": width, "step": stride, "n_windows": len(rows)},
    )


def bootstrap_jackknife(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    fit_func: Callable[[pd.DataFrame, pd.Series], Any] | None = None,
    n_replications: int = 50,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Bootstrap or jackknife-style uncertainty summary for importance."""

    frame = _as_feature_frame(X)
    target = pd.Series(np.asarray(y, dtype=float).reshape(-1), index=frame.index)
    if len(frame) != len(target):
        raise ValueError("X and y must have the same number of rows")
    if n_replications <= 0:
        raise ValueError("n_replications must be positive")
    rng = np.random.default_rng(random_state)
    values: dict[str, list[float]] = {str(column): [] for column in frame.columns}
    mode = "refit_coefficients" if fit_func is not None else "fixed_model_permutation"
    for _ in range(int(n_replications)):
        sample_pos = rng.integers(0, len(frame), size=len(frame))
        sample_x = frame.iloc[sample_pos]
        sample_y = target.iloc[sample_pos]
        if fit_func is not None:
            fit = fit_func(sample_x, sample_y)
            table = linear_coefficients(fit, sort=False)
            series = table.set_index("feature")["abs_coefficient"]
        else:
            table = permutation_importance(
                model,
                sample_x,
                sample_y,
                n_repeats=1,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            series = table.set_index("feature")["importance"]
        for feature in values:
            values[feature].append(float(series.get(feature, 0.0)))
    rows = []
    for feature, draws in values.items():
        arr = np.asarray(draws, dtype=float)
        rows.append(
            {
                "feature": feature,
                "importance": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "lower": float(np.quantile(arr, 0.05)),
                "upper": float(np.quantile(arr, 0.95)),
                "n_replications": int(n_replications),
            }
        )
    return _attach_schema(
        pd.DataFrame(rows).sort_values("importance", ascending=False, kind="stable").reset_index(drop=True),
        kind="bootstrap_jackknife",
        model=model,
        method=mode,
        n_features=frame.shape[1],
        metadata={"n_replications": int(n_replications), "mode": mode},
    )


def gradient_attribution(
    model: Any,
    X: pd.DataFrame,
    *,
    method: str = "saliency_map",
    baseline: float | pd.DataFrame | np.ndarray | None = None,
    n_steps: int = 50,
    n_samples: int = 20,
    noise_scale: float = 0.0,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Gradient attribution for torch-backed models."""

    key = str(method).lower().replace("-", "_")
    if key not in {"saliency_map", "integrated_gradients", "deep_lift", "gradient_shap"}:
        raise ValueError("method must be 'saliency_map', 'integrated_gradients', 'deep_lift', or 'gradient_shap'")
    torch, torch_model, tensor, feature_names = _torch_attribution_context(model, X)
    baseline_tensor = _baseline_tensor(torch, tensor, baseline)
    if key == "deep_lift":
        try:
            captum = import_module("captum.attr")
        except ImportError as exc:
            raise ImportError("deep_lift requires captum; install macroforecast[deep]") from exc
        attr_obj = captum.DeepLift(torch_model)
        attribution = attr_obj.attribute(tensor, baselines=baseline_tensor)
    elif key == "saliency_map":
        attribution = _torch_gradient(torch, torch_model, tensor)
    elif key == "integrated_gradients":
        attribution = _manual_integrated_gradients(
            torch,
            torch_model,
            tensor,
            baseline_tensor,
            n_steps=max(1, int(n_steps)),
        )
    else:
        attribution = _manual_gradient_shap(
            torch,
            torch_model,
            tensor,
            baseline_tensor,
            n_samples=max(1, int(n_samples)),
            noise_scale=float(noise_scale),
            random_state=random_state,
        )
    values = attribution.detach().cpu().numpy()
    if values.ndim == 3:
        # Recurrent models: aggregate over sequence positions for each feature.
        feature_values = values.mean(axis=1)
    else:
        feature_values = values
    rows = []
    for pos, feature in enumerate(feature_names):
        column = feature_values[:, pos]
        rows.append(
            {
                "feature": str(feature),
                "importance": float(np.mean(np.abs(column))),
                "mean_attribution": float(np.mean(column)),
                "std_attribution": float(np.std(column, ddof=1)) if len(column) > 1 else 0.0,
                "method": key,
            }
        )
    return _attach_schema(
        pd.DataFrame(rows).sort_values("importance", ascending=False, kind="stable").reset_index(drop=True),
        kind=key,
        model=model,
        method=key,
        n_features=len(feature_names),
        metadata={
            "n_obs": int(_as_feature_frame(X).shape[0]),
            "n_steps": int(n_steps),
            "n_samples": int(n_samples),
            "noise_scale": float(noise_scale),
        },
    )


def saliency_map(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    """Vanilla input-gradient attribution for torch-backed models."""

    return gradient_attribution(model, X, method="saliency_map")


def integrated_gradients(
    model: Any,
    X: pd.DataFrame,
    *,
    baseline: float | pd.DataFrame | np.ndarray | None = None,
    n_steps: int = 50,
) -> pd.DataFrame:
    """Integrated gradients for torch-backed models."""

    return gradient_attribution(
        model,
        X,
        method="integrated_gradients",
        baseline=baseline,
        n_steps=n_steps,
    )


def gradient_shap(
    model: Any,
    X: pd.DataFrame,
    *,
    baseline: float | pd.DataFrame | np.ndarray | None = None,
    n_samples: int = 20,
    noise_scale: float = 0.0,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Expected-gradients approximation to GradientSHAP."""

    return gradient_attribution(
        model,
        X,
        method="gradient_shap",
        baseline=baseline,
        n_samples=n_samples,
        noise_scale=noise_scale,
        random_state=random_state,
    )


def deep_lift(
    model: Any,
    X: pd.DataFrame,
    *,
    baseline: float | pd.DataFrame | np.ndarray | None = None,
) -> pd.DataFrame:
    """DeepLift attribution through Captum for torch-backed models."""

    return gradient_attribution(model, X, method="deep_lift", baseline=baseline)


def lstm_hidden_state(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    """LSTM/GRU hidden-unit activation importance for torch-backed models."""

    estimator = _estimator(model)
    if getattr(estimator, "kind", None) == "transformer":
        raise NotImplementedError("lstm_hidden_state is only defined for LSTM/GRU models")
    try:
        torch = import_module("torch")
    except ImportError as exc:
        raise ImportError("lstm_hidden_state requires torch; install macroforecast[deep]") from exc
    torch_model = getattr(estimator, "model_", None) or getattr(estimator, "_model", None)
    if torch_model is None:
        raise NotImplementedError("lstm_hidden_state requires a fitted torch model")
    rnn = getattr(torch_model, "rnn", None) or getattr(torch_model, "cell", None)
    if rnn is None:
        raise NotImplementedError("lstm_hidden_state requires a model with an LSTM/GRU recurrent cell")
    frame = _as_feature_frame(X)
    if getattr(estimator, "feature_names_in_", None):
        frame = frame.reindex(columns=list(estimator.feature_names_in_), fill_value=0.0)
    values = frame.astype(float).to_numpy(dtype=float)
    x_mean = getattr(estimator, "x_mean_", None)
    x_scale = getattr(estimator, "x_scale_", None)
    if x_mean is not None and x_scale is not None:
        values = (values - np.asarray(x_mean, dtype=float)) / np.asarray(x_scale, dtype=float)
    sequence_length = int(getattr(estimator, "sequence_length", 1))
    prefix = getattr(estimator, "train_tail_", None)
    prefix = np.empty((0, values.shape[1])) if prefix is None else np.asarray(prefix, dtype=float)
    combined = np.vstack([prefix, values])
    seq = np.stack([combined[i : i + sequence_length] for i in range(len(values))])
    captured: list[Any] = []

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        captured.append(output[0].detach().cpu())

    handle = rnn.register_forward_hook(hook)
    try:
        torch_model.eval()
        device = torch.device(getattr(estimator, "device_", "cpu") or "cpu")
        with torch.no_grad():
            torch_model(torch.tensor(seq, dtype=torch.float32, device=device))
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("lstm_hidden_state did not capture recurrent activations")
    out = captured[0]
    importance = out.abs().mean(dim=(0, 1)).detach().cpu().numpy()
    table = pd.DataFrame(
        {
            "feature": [f"hidden_unit_{i}" for i in range(len(importance))],
            "importance": [float(v) for v in importance],
            "coefficient": [None] * len(importance),
        }
    )
    return _attach_schema(
        table,
        kind="lstm_hidden_state",
        model=model,
        method="torch_recurrent_forward_hook",
        n_features=frame.shape[1],
        metadata={"hidden_size": int(len(importance)), "n_obs": int(len(frame))},
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


def _estimator(model: Any) -> Any:
    return model.estimator if isinstance(model, ModelFit) else model


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


def _safe_qcut(values: pd.Series, n_bins: int) -> pd.Series:
    try:
        bins = pd.qcut(values, q=int(n_bins), labels=False, duplicates="drop")
        return pd.Series(bins, index=values.index)
    except Exception:
        return pd.Series(np.zeros(len(values), dtype=int), index=values.index)


def _grid_values(values: pd.Series, grid_size: int) -> np.ndarray:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        raise ValueError("grid feature contains no finite values")
    quantiles = np.linspace(0.0, 1.0, int(grid_size))
    grid = np.unique(np.quantile(clean.to_numpy(dtype=float), quantiles))
    if len(grid) == 1:
        return np.repeat(grid[0], int(grid_size))
    return grid


def _select_row(frame: pd.DataFrame, row: int | str | pd.Timestamp) -> pd.Series:
    if isinstance(row, int):
        selected = frame.iloc[row].copy()
    else:
        selected = frame.loc[row].copy()
    if isinstance(selected, pd.DataFrame):
        selected = selected.iloc[0]
    return selected


def _jsonish_index(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    residual = np.asarray(y, dtype=float) - np.asarray(pred, dtype=float)
    denom = float(np.sum((y - np.mean(y)) ** 2))
    if denom <= 1e-15:
        return 0.0
    return float(1.0 - np.sum(residual**2) / denom)


def _infer_importance_column(frame: pd.DataFrame) -> str:
    for column in ("importance", "abs_contribution", "abs_coefficient", "contribution", "coefficient"):
        if column in frame.columns:
            return column
    raise ValueError("table must contain an importance-like column")


def _normalize_group_mapping(groups: Mapping[str, str | Sequence[str]] | None) -> dict[str, str]:
    if groups is None:
        return {}
    mapping: dict[str, str] = {}
    for key, value in groups.items():
        if isinstance(value, str):
            mapping[str(key)] = value
        else:
            for feature in value:
                mapping[str(feature)] = str(key)
    return mapping


def _aggregate_importance(
    frame: pd.DataFrame,
    *,
    group_by: str,
    value_column: str,
    aggregation: str,
) -> pd.DataFrame:
    if aggregation == "sum":
        grouped = frame.groupby(group_by, as_index=False)[value_column].sum()
    elif aggregation == "mean":
        grouped = frame.groupby(group_by, as_index=False)[value_column].mean()
    elif aggregation == "max_abs":
        grouped = (
            frame.assign(__abs__=frame[value_column].abs())
            .groupby(group_by, as_index=False)["__abs__"]
            .max()
            .rename(columns={"__abs__": value_column})
        )
    elif aggregation == "signed_sum":
        grouped = frame.groupby(group_by, as_index=False)[value_column].sum()
    else:
        raise ValueError("aggregation must be 'sum', 'mean', 'max_abs', or 'signed_sum'")
    grouped = grouped.rename(columns={value_column: "importance"})
    return grouped.sort_values("importance", ascending=False, kind="stable").reset_index(drop=True)


def _first_present(frame: pd.DataFrame, columns: Sequence[str]) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None


def _pipeline_loss_contribution(values: np.ndarray, *, method: str) -> np.ndarray:
    n_items = len(values)
    if method == "marginal_addition":
        return float(np.max(values)) - values
    if method == "leave_one_out_pipeline":
        full = float(np.mean(values))
        out = np.zeros(n_items, dtype=float)
        for idx in range(n_items):
            without = np.delete(values, idx)
            out[idx] = float(np.mean(without) - full) if len(without) else 0.0
        return out
    shapley = np.zeros(n_items, dtype=float)
    indices = list(range(n_items))
    for size in range(n_items):
        for subset in combinations(indices, size):
            subset_set = set(subset)
            subset_loss = float(np.mean(values[list(subset)])) if subset else 0.0
            weight = 1.0 / (n_items * comb(n_items - 1, size))
            for idx in indices:
                if idx in subset_set:
                    continue
                new_subset = list(subset) + [idx]
                new_loss = float(np.mean(values[new_subset]))
                shapley[idx] += weight * (subset_loss - new_loss if subset else -new_loss)
    return shapley


def _design_matrix(frame: pd.DataFrame, *, add_intercept: bool) -> np.ndarray:
    matrix = frame.to_numpy(dtype=float)
    if add_intercept:
        matrix = np.column_stack([np.ones(len(frame), dtype=float), matrix])
    return matrix


def _var_results(model: Any) -> Any:
    estimator = _estimator(model)
    candidates = [
        estimator,
        getattr(estimator, "_results", None),
        getattr(getattr(estimator, "_var", None), "_results", None),
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "irf"):
            return candidate
    raise ValueError("model does not expose fitted VAR results")


def _var_names(results: Any) -> list[str]:
    names = getattr(results, "names", None)
    if names is not None:
        return [str(name) for name in names]
    endog_names = getattr(results, "endog_names", None)
    if endog_names is not None:
        if isinstance(endog_names, str):
            return [endog_names]
        return [str(name) for name in endog_names]
    k = int(np.asarray(getattr(results, "sigma_u", np.eye(1))).shape[0])
    return [f"var_{idx}" for idx in range(k)]


def _target_position(names: Sequence[str], target: str | int | None) -> int:
    if target is None:
        return 0
    if isinstance(target, int):
        if target < 0 or target >= len(names):
            raise ValueError("target index is out of range")
        return int(target)
    if str(target) not in names:
        raise ValueError(f"target {target!r} is not in VAR names")
    return list(names).index(str(target))


def _var_irf_table(
    model: Any,
    *,
    n_periods: int,
    target: str | int | None,
    method: str,
) -> pd.DataFrame:
    if n_periods < 0:
        raise ValueError("n_periods must be non-negative")
    results = _var_results(model)
    names = _var_names(results)
    target_pos = _target_position(names, target)
    sigma = np.asarray(getattr(results, "sigma_u", np.eye(len(names))), dtype=float)
    if method == "generalized_irf":
        k = sigma.shape[0]
        try:
            irf_obj = results.irf(int(n_periods), var_decomp=np.eye(k))
        except Exception:
            irf_obj = results.irf(int(n_periods))
        irfs = np.asarray(irf_obj.irfs, dtype=float)
        rows: list[dict[str, Any]] = []
        for shock_pos, name in enumerate(names):
            e_j = np.zeros(k, dtype=float)
            e_j[shock_pos] = 1.0
            sigma_jj = float(sigma[shock_pos, shock_pos])
            scale = 1.0 if sigma_jj <= 0 else sigma_jj ** -0.5
            response = 0.0
            for horizon in range(irfs.shape[0]):
                girf = scale * irfs[horizon] @ sigma @ e_j
                response += abs(float(girf[target_pos]))
            rows.append({"feature": str(name), "importance": response, "coefficient": None, "status": "operational"})
    else:
        irf_obj = results.irf(int(n_periods))
        values = np.asarray(getattr(irf_obj, "orth_irfs", irf_obj.irfs), dtype=float)
        rows = [
            {
                "feature": str(name),
                "importance": float(np.sum(np.abs(values[:, target_pos, shock_pos]))),
                "coefficient": None,
                "status": "operational",
            }
            for shock_pos, name in enumerate(names)
        ]
    return _attach_schema(
        pd.DataFrame(rows).sort_values("importance", ascending=False, kind="stable").reset_index(drop=True),
        kind=method,
        model=model,
        method=method,
        n_features=len(names),
        metadata={"n_periods": int(n_periods), "target": names[target_pos]},
    )


def _mrf_beta_names(estimator: Any, output: Mapping[str, Any], n_columns: int) -> list[str]:
    raw_names: list[str] = []
    yandx = output.get("YandX")
    if isinstance(yandx, pd.DataFrame) and yandx.shape[1] >= 2:
        raw_names = [str(column) for column in yandx.columns[1:]]
    elif getattr(estimator, "x_columns", None):
        raw_names = [str(column) for column in estimator.x_columns]
    elif getattr(estimator, "_feature_names", None):
        raw_names = [str(column) for column in estimator._feature_names]
    names = ["__intercept__", *raw_names]
    if len(names) < n_columns:
        names.extend(f"beta_{idx}" for idx in range(len(names), n_columns))
    return names[:n_columns]


def _torch_attribution_context(model: Any, X: pd.DataFrame) -> tuple[Any, Any, Any, list[str]]:
    try:
        torch = import_module("torch")
    except ImportError as exc:
        raise ImportError("gradient attribution requires torch; install macroforecast[deep]") from exc
    estimator = _estimator(model)
    torch_model = getattr(estimator, "model_", None)
    if torch_model is None:
        raise NotImplementedError("gradient attribution requires a fitted torch-backed model")
    frame = _as_feature_frame(X)
    feature_names = list(getattr(estimator, "feature_names_in_", ())) or [str(column) for column in frame.columns]
    frame = frame.reindex(columns=feature_names, fill_value=0.0).astype(float)
    values = frame.to_numpy(dtype=float)
    x_mean = getattr(estimator, "x_mean_", None)
    x_scale = getattr(estimator, "x_scale_", None)
    if x_mean is not None and x_scale is not None:
        values = (values - np.asarray(x_mean, dtype=float)) / np.asarray(x_scale, dtype=float)
    device = torch.device(getattr(estimator, "device_", "cpu") or "cpu")
    kind = getattr(estimator, "kind", None)
    if kind in {"lstm", "gru", "transformer"}:
        sequence_length = int(getattr(estimator, "sequence_length", 1))
        prefix = getattr(estimator, "train_tail_", None)
        prefix = np.empty((0, values.shape[1])) if prefix is None else np.asarray(prefix, dtype=float)
        combined = np.vstack([prefix, values])
        tensor_values = np.stack([combined[i : i + sequence_length] for i in range(len(values))])
    else:
        tensor_values = values
    tensor = torch.tensor(tensor_values, dtype=torch.float32, device=device, requires_grad=True)
    torch_model.eval()
    return torch, torch_model, tensor, feature_names


def _baseline_tensor(torch: Any, tensor: Any, baseline: float | pd.DataFrame | np.ndarray | None) -> Any:
    if baseline is None:
        return torch.zeros_like(tensor)
    if isinstance(baseline, (int, float)):
        return torch.full_like(tensor, float(baseline))
    arr = np.asarray(baseline, dtype=float)
    if arr.shape != tuple(tensor.shape):
        if arr.ndim == 2 and tensor.ndim == 3 and arr.shape == (tensor.shape[0], tensor.shape[2]):
            arr = np.repeat(arr[:, None, :], tensor.shape[1], axis=1)
        else:
            arr = np.broadcast_to(arr, tuple(tensor.shape))
    return torch.tensor(arr, dtype=tensor.dtype, device=tensor.device)


def _torch_model_output(model: Any, tensor: Any) -> Any:
    output = model(tensor)
    if isinstance(output, tuple):
        output = output[0]
    return output.reshape(-1).sum()


def _torch_gradient(torch: Any, model: Any, tensor: Any) -> Any:
    point = tensor.detach().clone().requires_grad_(True)
    output = _torch_model_output(model, point)
    grad = torch.autograd.grad(output, point, create_graph=False)[0]
    return grad


def _manual_integrated_gradients(
    torch: Any,
    model: Any,
    tensor: Any,
    baseline: Any,
    *,
    n_steps: int,
) -> Any:
    total = torch.zeros_like(tensor)
    for alpha in torch.linspace(0.0, 1.0, steps=n_steps, device=tensor.device):
        point = baseline + alpha * (tensor - baseline)
        total = total + _torch_gradient(torch, model, point)
    return (tensor - baseline) * total / float(n_steps)


def _manual_gradient_shap(
    torch: Any,
    model: Any,
    tensor: Any,
    baseline: Any,
    *,
    n_samples: int,
    noise_scale: float,
    random_state: int | None,
) -> Any:
    if random_state is not None:
        torch.manual_seed(int(random_state))
    total = torch.zeros_like(tensor)
    for _ in range(n_samples):
        alpha = torch.rand((), device=tensor.device)
        point = baseline + alpha * (tensor - baseline)
        if noise_scale > 0:
            point = point + torch.randn_like(point) * float(noise_scale)
        total = total + _torch_gradient(torch, model, point)
    return (tensor - baseline) * total / float(n_samples)


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
