from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from macroforecast.data import attach_metadata, validate_panel
from macroforecast.feature_engineering._sparse_ic import (
    select_sparse_ic_params,
    sparse_ic_metadata,
)
from macroforecast.feature_engineering.shared import (
    _coerce_input,
    _metadata_frame,
    _resolve_columns,
    _warn_if_full_sample_fit,
)
from macroforecast.feature_engineering.types import FeatureInput


@dataclass(frozen=True)
class FeatureSelectionResult:
    """Fitted column-selection result shared by direct and runner-safe APIs."""

    selected_columns: tuple[str, ...]
    scores: dict[str, float]
    method: str
    n_features: int | float | None
    resolved_n_features: int
    n_fit_rows: int
    fit_policy: str
    target_required: bool
    metadata: dict[str, Any]


def normalize_feature_selection_method(value: str) -> str:
    aliases = {
        "variance": "variance_selection",
        "variance_selection": "variance_selection",
        "var": "variance_selection",
        "correlation": "correlation_selection",
        "correlation_selection": "correlation_selection",
        "corr": "correlation_selection",
        "lasso": "lasso_selection",
        "lasso_selection": "lasso_selection",
        "boruta": "boruta_selection",
        "boruta_selection": "boruta_selection",
        "rfe": "rfe_selection",
        "rfecv": "rfe_selection",
        "rfe_selection": "rfe_selection",
        "recursive_feature_elimination": "rfe_selection",
        "lasso_path": "lasso_path_selection",
        "lasso_path_selection": "lasso_path_selection",
        "stability": "stability_selection",
        "stability_selection": "stability_selection",
        "genetic": "genetic_selection",
        "genetic_selection": "genetic_selection",
        "genetic_algorithm_selection": "genetic_selection",
    }
    key = str(value).lower()
    if key not in aliases:
        raise ValueError(
            "feature selection method must be one of "
            f"{sorted(aliases)}; got {value!r}"
        )
    return aliases[key]


def feature_selection_requires_target(method: str) -> bool:
    return normalize_feature_selection_method(method) != "variance_selection"


def select_features(
    source: pd.DataFrame,
    target: pd.Series | None = None,
    *,
    n_features: int | float | None = 0.5,
    method: str = "variance_selection",
    min_train_size: int | None = None,
    random_state: int | None = 0,
    **params: Any,
) -> FeatureSelectionResult:
    """Fit one feature-selection rule on a training matrix."""

    if source.empty:
        raise ValueError("feature selection requires at least one source column")
    method_value = normalize_feature_selection_method(method)
    columns = tuple(str(column) for column in source.columns)
    x_source = source.astype(float)
    n_keep = _resolve_keep_count(n_features, n_columns=len(columns))
    target_required = feature_selection_requires_target(method_value)
    if target_required:
        if target is None:
            raise ValueError(f"{method_value} feature selection requires a target")
        joined = pd.concat([x_source, target.rename("__target__")], axis=1).dropna()
        min_rows = _resolve_min_train_size(min_train_size, minimum=2)
        if len(joined) < min_rows:
            raise ValueError(
                f"{method_value} feature selection requires at least "
                f"{min_rows} target-aligned complete rows"
            )
        x_train = joined.loc[:, columns]
        y_train = joined["__target__"].astype(float)
        fit_policy = "fixed_fit_panel_target_aligned_rows"
    else:
        x_train = x_source
        y_train = None
        min_rows = 1 if min_train_size is None else int(min_train_size)
        n_complete = int(len(x_source.dropna()))
        if n_complete < min_rows:
            raise ValueError(
                f"variance feature selection requires at least {min_rows} complete rows"
            )
        fit_policy = "fixed_fit_panel_columns"

    if method_value == "variance_selection":
        scores = x_train.var(axis=0, skipna=True).fillna(0.0).abs()
        metadata = {"score": "sample_variance"}
    elif method_value == "correlation_selection":
        scores = x_train.corrwith(y_train).abs().fillna(0.0)  # type: ignore[arg-type]
        metadata = {"score": "absolute_target_correlation"}
    elif method_value == "lasso_selection":
        scores, metadata = _lasso_scores(x_train, y_train, params=params)
    elif method_value == "lasso_path_selection":
        scores, metadata = _lasso_path_scores(x_train, y_train, params=params)
    elif method_value == "rfe_selection":
        scores, metadata = _rfe_scores(
            x_train,
            y_train,
            n_keep=max(1, n_keep or 1),
            params=params,
            random_state=random_state,
        )
    elif method_value == "boruta_selection":
        scores, metadata = _boruta_scores(
            x_train,
            y_train,
            params=params,
            random_state=random_state,
        )
    elif method_value == "stability_selection":
        scores, metadata = _stability_scores(
            x_train,
            y_train,
            params=params,
            random_state=random_state,
        )
    elif method_value == "genetic_selection":
        resolved = max(1, n_keep or int(np.ceil(len(columns) * 0.5)))
        scores, metadata = _genetic_scores(
            x_train,
            y_train,
            n_keep=min(resolved, len(columns)),
            params=params,
            random_state=random_state,
        )
    else:  # pragma: no cover - guarded by normalizer.
        raise ValueError(f"unsupported feature selection method {method_value!r}")
    if params:
        raise ValueError(f"unsupported {method_value} parameter(s): {sorted(params)}")

    scores = scores.reindex(columns).fillna(0.0).astype(float)
    selected = _selected_columns(scores, n_keep=n_keep, method_metadata=metadata)
    return FeatureSelectionResult(
        selected_columns=selected,
        scores={str(key): float(value) for key, value in scores.items()},
        method=method_value,
        n_features=n_features,
        resolved_n_features=len(selected),
        n_fit_rows=int(len(x_train.dropna())) if y_train is None else int(len(x_train)),
        fit_policy=fit_policy,
        target_required=target_required,
        metadata=metadata,
    )


def variance_selection(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    n_features: int | float = 0.5,
    min_train_size: int | None = None,
) -> pd.DataFrame:
    """Select columns with the largest sample variance."""

    return _selection_features(
        data,
        None,
        metadata=metadata,
        columns=columns,
        n_features=n_features,
        method="variance_selection",
        min_train_size=min_train_size,
    )


def correlation_selection(
    data: FeatureInput,
    target: str | pd.Series | None = None,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    n_features: int | float = 0.5,
    min_train_size: int | None = None,
    warn_full_sample: bool = True,
) -> pd.DataFrame:
    """Select columns with the largest absolute target correlation."""

    return _selection_features(
        data,
        target,
        metadata=metadata,
        columns=columns,
        n_features=n_features,
        method="correlation_selection",
        min_train_size=min_train_size,
        warn_full_sample=warn_full_sample,
    )


def lasso_selection(
    data: FeatureInput,
    target: str | pd.Series | None = None,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    n_features: int | float = 0.5,
    alpha: float = 0.001,
    lambda_search: Any | None = None,
    min_train_size: int | None = None,
    warn_full_sample: bool = True,
) -> pd.DataFrame:
    """Select columns by absolute lasso coefficient magnitude."""

    return _selection_features(
        data,
        target,
        metadata=metadata,
        columns=columns,
        n_features=n_features,
        method="lasso_selection",
        min_train_size=min_train_size,
        warn_full_sample=warn_full_sample,
        alpha=alpha,
        lambda_search=lambda_search,
    )


def lasso_path_selection(
    data: FeatureInput,
    target: str | pd.Series | None = None,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    n_features: int | float = 0.5,
    eps: float = 1e-3,
    n_alphas: int = 100,
    normalize_features: bool = True,
    positive: bool = False,
    min_train_size: int | None = None,
    warn_full_sample: bool = True,
) -> pd.DataFrame:
    """Select columns by lasso-path inclusion frequency."""

    return _selection_features(
        data,
        target,
        metadata=metadata,
        columns=columns,
        n_features=n_features,
        method="lasso_path_selection",
        min_train_size=min_train_size,
        warn_full_sample=warn_full_sample,
        eps=eps,
        n_alphas=n_alphas,
        normalize_features=normalize_features,
        positive=positive,
    )


def rfe_selection(
    data: FeatureInput,
    target: str | pd.Series | None = None,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    n_features: int | float = 0.5,
    estimator: str = "ridge",
    step: int | float = 1,
    use_cv: bool = False,
    cv_folds: int = 5,
    min_train_size: int | None = None,
    random_state: int | None = 0,
    warn_full_sample: bool = True,
) -> pd.DataFrame:
    """Select columns by recursive feature elimination."""

    return _selection_features(
        data,
        target,
        metadata=metadata,
        columns=columns,
        n_features=n_features,
        method="rfe_selection",
        min_train_size=min_train_size,
        random_state=random_state,
        warn_full_sample=warn_full_sample,
        estimator=estimator,
        step=step,
        use_cv=use_cv,
        cv_folds=cv_folds,
    )


def boruta_selection(
    data: FeatureInput,
    target: str | pd.Series | None = None,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    n_features: int | float = 0.5,
    n_estimators: int = 100,
    max_iter: int = 100,
    alpha: float = 0.05,
    include_tentative: bool = False,
    max_depth: int | None = None,
    min_train_size: int | None = None,
    random_state: int | None = 0,
    warn_full_sample: bool = True,
) -> pd.DataFrame:
    """Select columns using a Boruta-style shadow-feature test."""

    return _selection_features(
        data,
        target,
        metadata=metadata,
        columns=columns,
        n_features=n_features,
        method="boruta_selection",
        min_train_size=min_train_size,
        random_state=random_state,
        warn_full_sample=warn_full_sample,
        n_estimators=n_estimators,
        max_iter=max_iter,
        alpha=alpha,
        include_tentative=include_tentative,
        max_depth=max_depth,
    )


def stability_selection(
    data: FeatureInput,
    target: str | pd.Series | None = None,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    n_features: int | float | None = 0.5,
    n_subsamples: int = 100,
    subsample_fraction: float = 0.5,
    pi_threshold: float = 0.6,
    base_estimator: str = "lasso",
    alpha: float = 0.01,
    l1_ratio: float = 0.5,
    min_train_size: int | None = None,
    random_state: int | None = 0,
    warn_full_sample: bool = True,
) -> pd.DataFrame:
    """Select columns by repeated sparse-model subsampling frequency."""

    return _selection_features(
        data,
        target,
        metadata=metadata,
        columns=columns,
        n_features=n_features,
        method="stability_selection",
        min_train_size=min_train_size,
        random_state=random_state,
        warn_full_sample=warn_full_sample,
        n_subsamples=n_subsamples,
        subsample_fraction=subsample_fraction,
        pi_threshold=pi_threshold,
        base_estimator=base_estimator,
        alpha=alpha,
        l1_ratio=l1_ratio,
    )


def genetic_selection(
    data: FeatureInput,
    target: str | pd.Series | None = None,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    n_features: int | float = 0.5,
    population_size: int = 30,
    n_generations: int = 50,
    crossover_prob: float = 0.8,
    mutation_prob: float | None = None,
    fitness_estimator: str = "ridge",
    cv_folds: int = 3,
    min_train_size: int | None = None,
    random_state: int | None = 0,
    warn_full_sample: bool = True,
) -> pd.DataFrame:
    """Select columns using a small genetic subset search."""

    params: dict[str, Any] = {
        "population_size": population_size,
        "n_generations": n_generations,
        "crossover_prob": crossover_prob,
        "fitness_estimator": fitness_estimator,
        "cv_folds": cv_folds,
    }
    if mutation_prob is not None:
        params["mutation_prob"] = mutation_prob
    return _selection_features(
        data,
        target,
        metadata=metadata,
        columns=columns,
        n_features=n_features,
        method="genetic_selection",
        min_train_size=min_train_size,
        random_state=random_state,
        warn_full_sample=warn_full_sample,
        **params,
    )


def _selection_features(
    data: FeatureInput,
    target: str | pd.Series | None,
    *,
    metadata: Mapping[str, Any] | None,
    columns: Iterable[str] | None,
    n_features: int | float | None,
    method: str,
    min_train_size: int | None = None,
    random_state: int | None = 0,
    warn_full_sample: bool = True,
    **params: Any,
) -> pd.DataFrame:
    base = _coerce_input(data, metadata=metadata)
    panel = base.panel
    validate_panel(panel)
    selected = _resolve_columns(panel, columns=columns)
    source = panel.loc[:, selected].astype(float)
    target_name = None
    target_series = None
    if method != "variance_selection":
        target_name, target_series = _resolve_target_argument(
            base,
            panel,
            target,
            context=f"{method}()",
        )
        _warn_if_full_sample_fit("full_sample", context=f"{method}()", enabled=warn_full_sample)
    selection = select_features(
        source,
        target_series,
        n_features=n_features,
        method=method,
        min_train_size=min_train_size,
        random_state=random_state,
        **params,
    )
    result = source.loc[:, selection.selected_columns].copy()
    result.index.name = "date"
    result.attrs["macroforecast_metadata"] = attach_metadata(
        base.metadata,
        f"feature_engineering_{method}",
        {
            "target": None if target_name is None else str(target_name),
            "columns": list(selected),
            "selected_columns": list(selection.selected_columns),
            "n_features": n_features,
            "resolved_n_features": int(selection.resolved_n_features),
            "method": method,
            "selection_params": selection.metadata,
            "fit_policy": selection.fit_policy.replace("fixed_fit_panel", "full_input"),
            "warn_full_sample": bool(warn_full_sample),
        },
    )
    result.attrs["macroforecast_feature_metadata"] = _metadata_frame(
        {
            "feature": str(column),
            "operation": method,
            "source": str(column),
            "parameter": f"rank={rank};score={selection.scores.get(column)}",
            "fit_policy": selection.fit_policy.replace("fixed_fit_panel", "full_input"),
            "inputs": ",".join(selected),
            "included": True,
            "target": target_name,
            "score": selection.scores.get(column),
        }
        for rank, column in enumerate(selection.selected_columns, start=1)
    )
    return result


def _resolve_target_argument(
    base: Any,
    panel: pd.DataFrame,
    target: str | pd.Series | None,
    *,
    context: str,
) -> tuple[str, pd.Series]:
    if isinstance(target, str):
        if target not in panel.columns:
            raise ValueError(f"target {target!r} is not in the panel")
        return target, panel[target].astype(float)
    if isinstance(target, pd.Series):
        name = str(target.name) if target.name is not None else "target"
        return name, target.astype(float)
    if base.target is not None:
        if base.target not in panel.columns:
            raise ValueError(f"target {base.target!r} is not in the panel")
        return str(base.target), panel[str(base.target)].astype(float)
    raise ValueError(f"{context} requires target or input target metadata")


def _lasso_scores(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    params: dict[str, Any],
) -> tuple[pd.Series, dict[str, Any]]:
    from sklearn.linear_model import Lasso

    alpha = float(params.pop("lasso_alpha", params.pop("alpha", 0.001)))
    max_iter = int(params.pop("max_iter", 20000))
    lambda_search = params.pop("lambda_search", None)
    metadata: dict[str, Any] = {"score": "absolute_lasso_coefficient"}
    if lambda_search is not None:
        result = select_sparse_ic_params(
            "lasso",
            X,
            y,
            lambda_search,
            allowed_params={"alpha"},
            fixed_params={"max_iter": max_iter},
        )
        selected = dict(result.best_params)
        alpha = float(selected["alpha"])
        metadata["lambda_selection"] = sparse_ic_metadata(result)
    model = Lasso(alpha=alpha, max_iter=max_iter)
    model.fit(X.to_numpy(dtype=float), y.to_numpy(dtype=float))
    scores = pd.Series(np.abs(model.coef_), index=X.columns, dtype=float)
    if float(scores.sum()) <= 1e-12:
        scores = X.corrwith(y).abs().fillna(0.0)
    metadata["alpha"] = alpha
    return scores, metadata


def _lasso_path_scores(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    params: dict[str, Any],
) -> tuple[pd.Series, dict[str, Any]]:
    from sklearn.linear_model import lasso_path

    eps = float(params.pop("eps", 1e-3))
    n_alphas = int(params.pop("n_alphas", 100))
    positive = bool(params.pop("positive", False))
    normalize = bool(params.pop("normalize_features", True))
    values = X.to_numpy(dtype=float)
    target = y.to_numpy(dtype=float)
    if normalize:
        values = _standardize(values)
        target = target - float(np.nanmean(target))
    _, coefs, _ = lasso_path(
        values,
        target,
        eps=eps,
        n_alphas=n_alphas,
        positive=positive,
    )
    nonzero_rate = (np.abs(coefs) > 1e-10).mean(axis=1)
    max_abs = np.abs(coefs).max(axis=1)
    if max_abs.max(initial=0.0) > 0:
        max_abs = max_abs / max_abs.max()
    scores = pd.Series(nonzero_rate + 1e-6 * max_abs, index=X.columns, dtype=float)
    if float(scores.sum()) <= 1e-12:
        scores = X.corrwith(y).abs().fillna(0.0)
    return scores, {
        "score": "lasso_path_inclusion_rate",
        "eps": eps,
        "n_alphas": n_alphas,
        "positive": positive,
        "normalize_features": normalize,
    }


def _rfe_scores(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_keep: int,
    params: dict[str, Any],
    random_state: int | None,
) -> tuple[pd.Series, dict[str, Any]]:
    from sklearn.feature_selection import RFE, RFECV
    from sklearn.model_selection import TimeSeriesSplit

    estimator_name = str(params.pop("estimator", "ridge"))
    step = params.pop("step", 1)
    use_cv = bool(params.pop("use_cv", False))
    cv_folds = int(params.pop("cv_folds", 5))
    estimator = _linear_estimator(estimator_name, params=params, random_state=random_state)
    if use_cv and len(X) >= 4 and cv_folds >= 2:
        n_splits = min(cv_folds, len(X) - 1)
        selector = RFECV(
            estimator=estimator,
            step=step,
            min_features_to_select=n_keep,
            cv=TimeSeriesSplit(n_splits=n_splits),
            scoring="neg_mean_squared_error",
        )
    else:
        selector = RFE(estimator=estimator, n_features_to_select=n_keep, step=step)
    selector.fit(X.to_numpy(dtype=float), y.to_numpy(dtype=float))
    ranking = pd.Series(selector.ranking_, index=X.columns, dtype=float)
    scores = 1.0 / ranking
    return scores, {
        "score": "inverse_rfe_rank",
        "estimator": estimator_name,
        "step": step,
        "use_cv": use_cv,
        "cv_folds": cv_folds,
    }


def _boruta_scores(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    params: dict[str, Any],
    random_state: int | None,
) -> tuple[pd.Series, dict[str, Any]]:
    from scipy.stats import binomtest
    from sklearn.ensemble import RandomForestRegressor

    n_estimators = int(params.pop("n_estimators_rf", params.pop("n_estimators", 100)))
    max_iter = int(params.pop("max_iter", 100))
    alpha = float(params.pop("alpha", 0.05))
    include_tentative = bool(params.pop("include_tentative", False))
    max_depth = params.pop("max_depth", None)
    rng = np.random.default_rng(random_state)
    hits = np.zeros(X.shape[1], dtype=float)
    importance_sum = np.zeros(X.shape[1], dtype=float)
    values = X.to_numpy(dtype=float)
    target = y.to_numpy(dtype=float)
    for iteration in range(max_iter):
        shadow = np.column_stack([rng.permutation(values[:, idx]) for idx in range(values.shape[1])])
        train = np.column_stack([values, shadow])
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=None if max_depth is None else int(max_depth),
            random_state=None if random_state is None else int(random_state) + iteration,
            n_jobs=1,
        )
        model.fit(train, target)
        importance = np.asarray(model.feature_importances_[: X.shape[1]], dtype=float)
        shadow_max = float(np.max(model.feature_importances_[X.shape[1] :]))
        hits += importance > shadow_max
        importance_sum += importance
    pvalues = np.array(
        [binomtest(int(hit), max_iter, 0.5, alternative="greater").pvalue for hit in hits],
        dtype=float,
    )
    hit_rate = hits / max(max_iter, 1)
    confirmed = (pvalues < alpha) & (hit_rate > 0.5)
    tentative = hit_rate > 0.5
    score_values = hit_rate + 1e-6 * _safe_unit_scale(importance_sum)
    if include_tentative:
        score_values = score_values + tentative.astype(float) * 1e-3
    score_values = score_values + confirmed.astype(float) * 1e-2
    return pd.Series(score_values, index=X.columns, dtype=float), {
        "score": "boruta_shadow_hit_rate",
        "n_estimators": n_estimators,
        "max_iter": max_iter,
        "alpha": alpha,
        "include_tentative": include_tentative,
        "confirmed": [str(col) for col, ok in zip(X.columns, confirmed, strict=True) if ok],
        "tentative": [str(col) for col, ok in zip(X.columns, tentative, strict=True) if ok],
    }


def _stability_scores(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    params: dict[str, Any],
    random_state: int | None,
) -> tuple[pd.Series, dict[str, Any]]:
    n_subsamples = int(params.pop("n_subsamples", 100))
    subsample_fraction = float(params.pop("subsample_fraction", 0.5))
    pi_threshold = float(params.pop("pi_thr", params.pop("pi_threshold", 0.6)))
    base_estimator = str(params.pop("base_estimator", "lasso"))
    alpha = float(params.pop("alpha", 0.01))
    l1_ratio = float(params.pop("l1_ratio", 0.5))
    rng = np.random.default_rng(random_state)
    n_obs = len(X)
    sample_size = max(2, min(n_obs, int(np.ceil(n_obs * subsample_fraction))))
    counts = np.zeros(X.shape[1], dtype=float)
    values = _standardize(X.to_numpy(dtype=float))
    target = y.to_numpy(dtype=float) - float(y.mean())
    for _ in range(n_subsamples):
        idx = np.sort(rng.choice(n_obs, size=sample_size, replace=False))
        model = _sparse_estimator(base_estimator, alpha=alpha, l1_ratio=l1_ratio)
        model.fit(values[idx], target[idx])
        counts += np.abs(np.asarray(model.coef_, dtype=float)) > 1e-10
    rates = counts / max(n_subsamples, 1)
    return pd.Series(rates, index=X.columns, dtype=float), {
        "score": "selection_frequency",
        "n_subsamples": n_subsamples,
        "subsample_fraction": subsample_fraction,
        "pi_threshold": pi_threshold,
        "base_estimator": base_estimator,
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "threshold_selected": [
            str(col) for col, rate in zip(X.columns, rates, strict=True) if rate >= pi_threshold
        ],
    }


def _genetic_scores(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_keep: int,
    params: dict[str, Any],
    random_state: int | None,
) -> tuple[pd.Series, dict[str, Any]]:
    population_size = int(params.pop("population_size", 30))
    n_generations = int(params.pop("n_generations", 50))
    crossover_prob = float(params.pop("crossover_prob", 0.8))
    mutation_prob = float(params.pop("mutation_prob", 1.0 / max(X.shape[1], 1)))
    fitness_estimator = str(params.pop("fitness_estimator", "ridge"))
    cv_folds = int(params.pop("cv_folds", 3))
    rng = np.random.default_rng(random_state)
    n_features = X.shape[1]
    population = np.vstack(
        [_random_mask(n_features, n_keep, rng) for _ in range(max(population_size, 2))]
    )
    values = _standardize(X.to_numpy(dtype=float))
    target = y.to_numpy(dtype=float)
    best_mask = population[0].copy()
    best_score = -np.inf
    for _ in range(max(n_generations, 1)):
        fitness = np.array(
            [
                _subset_fitness(
                    values,
                    target,
                    mask,
                    estimator_name=fitness_estimator,
                    cv_folds=cv_folds,
                    random_state=random_state,
                )
                for mask in population
            ],
            dtype=float,
        )
        winner = int(np.argmax(fitness))
        if fitness[winner] > best_score:
            best_score = float(fitness[winner])
            best_mask = population[winner].copy()
        next_population = [best_mask.copy()]
        while len(next_population) < len(population):
            parent_a = _tournament(population, fitness, rng)
            parent_b = _tournament(population, fitness, rng)
            if rng.random() < crossover_prob:
                child = np.where(rng.random(n_features) < 0.5, parent_a, parent_b)
            else:
                child = parent_a.copy()
            flips = rng.random(n_features) < mutation_prob
            child = np.logical_xor(child, flips)
            next_population.append(_repair_mask(child, n_keep, rng))
        population = np.vstack(next_population)
    scores = population.mean(axis=0)
    scores = np.maximum(scores, best_mask.astype(float))
    return pd.Series(scores, index=X.columns, dtype=float), {
        "score": "genetic_population_inclusion_rate",
        "population_size": population_size,
        "n_generations": n_generations,
        "crossover_prob": crossover_prob,
        "mutation_prob": mutation_prob,
        "fitness_estimator": fitness_estimator,
        "cv_folds": cv_folds,
        "best_score": best_score,
    }


def _selected_columns(
    scores: pd.Series,
    *,
    n_keep: int | None,
    method_metadata: dict[str, Any],
) -> tuple[str, ...]:
    if n_keep is None:
        threshold_selected = method_metadata.get("threshold_selected")
        if threshold_selected:
            return tuple(str(column) for column in threshold_selected if column in scores.index)
        positive = tuple(str(column) for column in scores.index[scores > 0])
        if positive:
            return positive
        n_keep = 1
    ranked = scores.sort_values(ascending=False, kind="mergesort")
    return tuple(str(column) for column in ranked.index[:n_keep])


def _resolve_keep_count(n_features: int | float | None, *, n_columns: int) -> int | None:
    if n_features is None:
        return None
    if isinstance(n_features, bool):
        raise TypeError("n_features must be an int, float fraction, or None")
    if isinstance(n_features, int):
        count = n_features
    else:
        value = float(n_features)
        if not 0 < value <= 1:
            raise ValueError("float n_features must be in (0, 1]")
        count = int(np.ceil(n_columns * value))
    if count <= 0:
        raise ValueError("n_features must select at least one column")
    if count > n_columns:
        raise ValueError("n_features cannot exceed the number of columns")
    return int(count)


def _resolve_min_train_size(value: int | None, *, minimum: int) -> int:
    if value is None:
        return minimum
    resolved = int(value)
    if resolved < minimum:
        raise ValueError(f"min_train_size must be >= {minimum}")
    return resolved


def _linear_estimator(name: str, *, params: dict[str, Any], random_state: int | None):
    from sklearn.linear_model import Lasso, LinearRegression, Ridge
    from sklearn.svm import LinearSVR

    key = name.lower()
    if key == "ridge":
        return Ridge(alpha=float(params.pop("alpha", 1.0)))
    if key == "lasso":
        return Lasso(alpha=float(params.pop("alpha", 0.001)), max_iter=20000)
    if key in {"ols", "linear_regression"}:
        return LinearRegression()
    if key in {"svr_linear", "linear_svr"}:
        return LinearSVR(
            C=float(params.pop("C", 1.0)),
            epsilon=float(params.pop("epsilon", 0.0)),
            random_state=random_state,
            max_iter=int(params.pop("max_iter", 20000)),
        )
    raise ValueError("estimator must be 'ridge', 'lasso', 'ols', or 'svr_linear'")


def _sparse_estimator(name: str, *, alpha: float, l1_ratio: float):
    from sklearn.linear_model import ElasticNet, Lasso

    key = name.lower()
    if key == "lasso":
        return Lasso(alpha=alpha, max_iter=20000)
    if key in {"elastic_net", "enet"}:
        return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000)
    raise ValueError("base_estimator must be 'lasso' or 'elastic_net'")


def _subset_fitness(
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    *,
    estimator_name: str,
    cv_folds: int,
    random_state: int | None,
) -> float:
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import TimeSeriesSplit

    selected = np.flatnonzero(mask)
    if selected.size == 0:
        return -np.inf
    values = X[:, selected]
    if len(values) < 4 or cv_folds < 2:
        split = max(1, int(len(values) * 0.75))
        splits = [(np.arange(split), np.arange(split, len(values)))]
    else:
        n_splits = min(cv_folds, len(values) - 1)
        splits = TimeSeriesSplit(n_splits=n_splits).split(values)
    losses: list[float] = []
    for train_idx, val_idx in splits:
        if len(val_idx) == 0:
            continue
        params: dict[str, Any] = {}
        estimator = _linear_estimator(estimator_name, params=params, random_state=random_state)
        estimator.fit(values[train_idx], y[train_idx])
        pred = estimator.predict(values[val_idx])
        losses.append(float(mean_squared_error(y[val_idx], pred)))
    return -float(np.mean(losses)) if losses else -np.inf


def _random_mask(n_features: int, n_keep: int, rng: np.random.Generator) -> np.ndarray:
    mask: np.ndarray = np.zeros(n_features, dtype=bool)
    mask[rng.choice(n_features, size=n_keep, replace=False)] = True
    return mask


def _repair_mask(mask: np.ndarray, n_keep: int, rng: np.random.Generator) -> np.ndarray:
    repaired = np.asarray(mask, dtype=bool).copy()
    on = np.flatnonzero(repaired)
    off = np.flatnonzero(~repaired)
    if len(on) > n_keep:
        repaired[rng.choice(on, size=len(on) - n_keep, replace=False)] = False
    elif len(on) < n_keep:
        repaired[rng.choice(off, size=n_keep - len(on), replace=False)] = True
    return repaired


def _tournament(population: np.ndarray, fitness: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    size = min(3, len(population))
    contenders = rng.choice(len(population), size=size, replace=False)
    return population[contenders[int(np.argmax(fitness[contenders]))]].copy()


def _standardize(values: np.ndarray) -> np.ndarray:
    center = np.nanmean(values, axis=0)
    scale = np.nanstd(values, axis=0, ddof=1)
    scale = np.where((~np.isfinite(scale)) | (scale == 0), 1.0, scale)
    return (values - center) / scale


def _safe_unit_scale(values: np.ndarray) -> np.ndarray:
    max_value = float(np.max(np.abs(values))) if values.size else 0.0
    if not np.isfinite(max_value) or max_value <= 0:
        return np.zeros_like(values, dtype=float)
    return values / max_value


__all__ = [
    "FeatureSelectionResult",
    "boruta_selection",
    "correlation_selection",
    "feature_selection_requires_target",
    "genetic_selection",
    "lasso_path_selection",
    "lasso_selection",
    "normalize_feature_selection_method",
    "rfe_selection",
    "select_features",
    "stability_selection",
    "variance_selection",
]
