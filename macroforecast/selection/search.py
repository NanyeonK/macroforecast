from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from macroforecast.evaluation import MetricLike, get_metric
from macroforecast.selection.builders import (
    _as_tuple,
    _candidates,
    _coerce_distribution,
    _normalize_method,
    _search_from_model,
    bayesian_search,
    choice,
    cv_path,
    fixed,
    genetic_search,
    grid,
    log_uniform,
    random_search,
    randint,
    search_spec,
    uniform,
)
from macroforecast.selection.optimizers import run_bayesian, run_genetic
from macroforecast.selection.runner import evaluate_candidate, parameter_columns, trial_frame
from macroforecast.selection.types import (
    SearchError,
    SearchResult,
    SearchSpec,
)
from macroforecast.models.specs import ModelSpec, get_model
from macroforecast.models.utils import align_xy, as_frame, as_series, resolve_xy
from macroforecast.window import Split, WindowSpec, resolve_window


def select_params(
    model: str | Callable[..., Any] | ModelSpec,
    X: Any,
    y: Any | None = None,
    search: SearchSpec | None = None,
    *,
    window: WindowSpec | str | None = None,
    splits: Sequence[tuple[Any, Any]] | None = None,
    metric: MetricLike = "mse",
    maximize: bool = False,
    fixed_params: dict[str, Any] | None = None,
    preset: str | None = None,
    method: str | None = None,
    random_state: int | None = None,
    n_iter: int | None = None,
    population_size: int | None = None,
    generations: int | None = None,
    mutation_rate: float | None = None,
) -> SearchResult:
    """Select model parameters by temporal validation.

    ``model`` can be a model name, a ``ModelSpec``, or a callable such as
    ``macroforecast.models.ridge`` that returns an object with ``predict(X)``.
    Registered models own their default parameters and hyperparameter spaces.
    This function evaluates parameter candidates. Validation timing can be
    supplied either as a window spec or as explicit integer-position splits
    produced by ``macroforecast.window``.
    """

    fit_model, model_spec = _resolve_model(model, preset=preset)
    frame, target = _resolve_selection_xy(model_spec, X, y, fixed_params=fixed_params)
    fit_model = _selection_fit_callable(fit_model, model_spec)
    if search is not None and _has_search_overrides(
        method=method,
        random_state=random_state,
        n_iter=n_iter,
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
    ):
        raise ValueError("search was provided, so method/random search options must be set on that SearchSpec")
    spec = search or (
        _search_from_model(
            model_spec,
            method=method,
            random_state=random_state,
            n_iter=n_iter,
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
        )
        if model_spec is not None
        else fixed(random_state=random_state)
    )
    spec = _prepare_search_spec(spec)
    metric_fn = get_metric(metric)
    validation_splits, split_name, split_metadata = _resolve_selection_splits(
        len(frame),
        window=window,
        splits=splits,
    )
    base_params = dict(fixed_params or {})
    rng = np.random.default_rng(spec.random_state)
    runtime_metadata: dict[str, Any] = {}

    if spec.method == "genetic":
        rows = run_genetic(
            fit_model,
            frame,
            target,
            validation_splits,
            metric_fn,
            spec,
            base_params,
            rng,
            maximize=maximize,
        )
    elif spec.method == "bayesian":
        rows, runtime_metadata = run_bayesian(
            fit_model,
            frame,
            target,
            validation_splits,
            metric_fn,
            spec,
            base_params,
            rng,
            maximize=maximize,
        )
    else:
        candidates = _candidates(spec, rng)
        rows = [
            evaluate_candidate(
                fit_model,
                frame,
                target,
                validation_splits,
                metric_fn,
                base_params,
                params,
                i,
            )
            for i, params in enumerate(candidates)
        ]

    trials = trial_frame(rows)
    ok = trials.loc[trials["status"] == "ok"].copy()
    if ok.empty:
        first_error = trials["error"].dropna().iloc[0] if "error" in trials else "unknown error"
        raise SearchError(f"All parameter trials failed: {first_error}", trials=trials)
    best_idx = ok["score"].idxmax() if maximize else ok["score"].idxmin()
    best_row = ok.loc[best_idx]
    best_params = {}
    for key in parameter_columns(trials):
        value = best_row[key]
        if isinstance(value, float) and np.isnan(value):
            continue
        best_params[key] = value
    return SearchResult(
        best_params=best_params,
        best_score=float(best_row["score"]),
        trials=trials,
        metric=metric,
        method=spec.method,
        window=split_name,
        metadata={
            "n_obs": len(frame),
            "n_splits": len(validation_splits),
            "maximize": maximize,
            **split_metadata,
            **_model_metadata(model_spec),
            **spec.metadata,
            **runtime_metadata,
        },
    )


def _resolve_model(
    model: str | Callable[..., Any] | ModelSpec,
    *,
    preset: str | None,
) -> tuple[Callable[..., Any], ModelSpec | None]:
    if isinstance(model, (str, ModelSpec)):
        model_spec = get_model(model, preset=preset)
        return model_spec.fit, model_spec
    if callable(model):
        try:
            model_spec = get_model(model, preset=preset)
        except ValueError:
            if preset is not None:
                raise
            return model, None
        return model_spec.fit, model_spec
    raise TypeError("model must be a model name, callable, or ModelSpec")


def _resolve_selection_xy(
    model_spec: ModelSpec | None,
    X: Any,
    y: Any | None,
    *,
    fixed_params: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    if model_spec is None:
        return resolve_xy(X, y)
    if model_spec.input_kind in {"target", "volatility"} and y is None:
        target = as_series(X)
        return pd.DataFrame(index=target.index), target
    if model_spec.input_kind == "panel":
        if y is None:
            frame = as_frame(X)
            target_column = _panel_target_column(model_spec, frame, fixed_params=fixed_params)
            temp_target = _temporary_target_name(frame.columns)
            target = frame[target_column].rename(temp_target)
            predictors = frame.drop(columns=[target_column])
            aligned = pd.concat([target, predictors], axis=1).dropna()
            resolved_target = aligned.pop(temp_target)
            resolved_target.name = str(target_column)
            return aligned, resolved_target
        return align_xy(as_frame(X), as_series(y))
    return resolve_xy(X, y)


def _selection_fit_callable(
    fit_model: Callable[..., Any],
    model_spec: ModelSpec | None,
) -> Callable[..., Any]:
    if model_spec is None or model_spec.input_kind != "panel":
        return fit_model

    def fit_panel_with_internal_target(X: Any, y: Any | None = None, **params: Any) -> Any:
        return model_spec.fit(X, y, **{**params, "target": None})

    return fit_panel_with_internal_target


def _panel_target_column(
    model_spec: ModelSpec,
    frame: pd.DataFrame,
    *,
    fixed_params: dict[str, Any] | None = None,
) -> Any:
    target = None
    if fixed_params is not None:
        target = fixed_params.get("target")
    if target is None:
        target = model_spec.all_params().get("target")
    if target is None:
        return frame.columns[0]
    if target not in frame.columns:
        raise ValueError(f"panel target column {target!r} is not present in X")
    return target


def _temporary_target_name(columns: pd.Index) -> str:
    name = "__macroforecast_target__"
    while name in columns:
        name = f"_{name}"
    return name


def _model_metadata(model_spec: ModelSpec | None) -> dict[str, Any]:
    if model_spec is None:
        return {}
    return {
        "model": model_spec.name,
        "model_family": model_spec.family,
        "model_preset": model_spec.preset,
    }


def _resolve_selection_splits(
    n_obs: int,
    *,
    window: WindowSpec | str | None,
    splits: Sequence[tuple[Any, Any]] | None,
) -> tuple[list[Split], str, dict[str, Any]]:
    if splits is not None and window is not None:
        raise ValueError("pass either window or splits, not both")
    if splits is None:
        window_spec = resolve_window(window)
        resolved = window_spec.split(n_obs)
        return (
            resolved,
            window_spec.to_dict()["method"],
            {
                "split_source": "window",
                "window": window_spec.to_dict(),
                "split_summary": _split_summary(resolved),
            },
        )
    resolved = _normalize_splits(splits, n_obs)
    return (
        resolved,
        "explicit_splits",
        {
            "split_source": "explicit",
            "window": None,
            "split_summary": _split_summary(resolved),
        },
    )


def _normalize_splits(splits: Sequence[tuple[Any, Any]], n_obs: int) -> list[Split]:
    if len(splits) == 0:
        raise ValueError("splits must contain at least one train/validation pair")
    resolved: list[Split] = []
    for split_id, pair in enumerate(splits):
        if len(pair) != 2:
            raise ValueError("each split must contain train and validation positions")
        train_idx = _normalize_split_positions(
            pair[0],
            n_obs=n_obs,
            split_id=split_id,
            side="train",
        )
        val_idx = _normalize_split_positions(
            pair[1],
            n_obs=n_obs,
            split_id=split_id,
            side="validation",
        )
        if np.intersect1d(train_idx, val_idx).size:
            raise ValueError(f"split {split_id} train and validation positions overlap")
        resolved.append((train_idx, val_idx))
    return resolved


def _normalize_split_positions(
    values: Any,
    *,
    n_obs: int,
    split_id: int,
    side: str,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"split {split_id} {side} positions must be one-dimensional")
    if arr.dtype == bool:
        if len(arr) != n_obs:
            raise ValueError(
                f"split {split_id} {side} boolean mask length must match n_obs"
            )
        arr = np.flatnonzero(arr)
    elif np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(int, copy=False)
    elif arr.dtype == object and all(_is_integer_position(value) for value in arr):
        arr = arr.astype(int)
    else:
        raise TypeError(f"split {split_id} {side} positions must be integer positions")
    if len(arr) == 0:
        raise ValueError(f"split {split_id} {side} positions must not be empty")
    if len(np.unique(arr)) != len(arr):
        raise ValueError(f"split {split_id} {side} positions must not contain duplicates")
    if int(arr.min()) < 0 or int(arr.max()) >= n_obs:
        raise ValueError(f"split {split_id} {side} positions are outside X/y bounds")
    return arr.astype(int, copy=False)


def _is_integer_position(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _split_summary(splits: list[Split]) -> list[dict[str, int]]:
    return [
        {
            "split": split_id,
            "n_train": int(len(train_idx)),
            "n_validation": int(len(val_idx)),
            "train_start_pos": int(train_idx.min()),
            "train_end_pos": int(train_idx.max()),
            "validation_start_pos": int(val_idx.min()),
            "validation_end_pos": int(val_idx.max()),
        }
        for split_id, (train_idx, val_idx) in enumerate(splits)
    ]


def _has_search_overrides(**values: Any) -> bool:
    return any(value is not None for value in values.values())


def _prepare_search_spec(spec: SearchSpec) -> SearchSpec:
    prepared = replace(spec)
    prepared.method = _normalize_method(prepared.method)
    prepared.param_grid = {key: _as_tuple(value) for key, value in prepared.param_grid.items()}
    prepared.param_distributions = {}
    for key, value in spec.param_distributions.items():
        try:
            prepared.param_distributions[key] = _coerce_distribution(value)
        except ValueError as exc:
            raise ValueError(f"invalid distribution for {key!r}: {exc}") from exc
    _validate_search_spec(prepared)
    return prepared


def _validate_search_spec(spec: SearchSpec) -> None:
    if spec.method not in {"fixed", "grid", "cv_path", "random", "bayesian", "genetic"}:
        raise ValueError(f"Unknown search method {spec.method!r}")
    if spec.n_iter < 1:
        raise ValueError("n_iter must be at least 1")
    if spec.population_size < 2:
        raise ValueError("population_size must be at least 2")
    if spec.generations < 1:
        raise ValueError("generations must be at least 1")
    if not 0 <= spec.mutation_rate <= 1:
        raise ValueError("mutation_rate must be between 0 and 1")
    if spec.method in {"random", "bayesian", "genetic"} and not spec.param_distributions:
        raise ValueError(f"{spec.method} search requires at least one parameter distribution")
    for key, values in spec.param_grid.items():
        if not values:
            raise ValueError(f"parameter grid for {key!r} cannot be empty")
    for key, dist in spec.param_distributions.items():
        try:
            dist.validate()
        except ValueError as exc:
            raise ValueError(f"invalid distribution for {key!r}: {exc}") from exc


__all__ = [
    "bayesian_search",
    "choice",
    "cv_path",
    "fixed",
    "genetic_search",
    "grid",
    "log_uniform",
    "random_search",
    "randint",
    "select_params",
    "search_spec",
    "uniform",
]
