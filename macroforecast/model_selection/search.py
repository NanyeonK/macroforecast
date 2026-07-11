from __future__ import annotations

import math

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from macroforecast.metrics import MetricLike, get_metric
from macroforecast.model_ensemble import get_model_ensemble
from macroforecast.model_selection.builders import (
    _as_tuple,
    _candidates,
    _coerce_distribution,
    _normalize_method,
    _search_from_model,
    bayesian_search,
    choice,
    custom_search,
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
from macroforecast.model_selection.optimizers import run_bayesian, run_genetic
from macroforecast.model_selection.runner import (
    evaluate_candidate,
    evaluate_candidate_group,
    parameter_columns,
    trial_frame,
)
from macroforecast.model_selection.splitters import _resolve_validation_splitter_with_fold_ids
from macroforecast.model_selection.types import (
    ScoreAggregation,
    SearchError,
    SearchResult,
    SearchSpec,
    SearchTrial,
    _normalize_score_aggregation,
)
from macroforecast.models.specs import ModelSpec, PrefixSearchSpec, get_model
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
    allow_non_temporal_splits: bool = False,
    score_aggregation: ScoreAggregation | None = None,
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
        raise ValueError(
            "search was provided, so method/random search options must be set on that SearchSpec"
        )
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
    score_aggregation_value = _resolve_score_aggregation(
        score_aggregation,
        spec.score_aggregation,
    )
    if score_aggregation_value != spec.score_aggregation:
        spec = replace(spec, score_aggregation=score_aggregation_value)
    if spec.method == "information_criterion":
        return select_by_information_criterion(
            model,
            X,
            y,
            search=spec,
            criterion=spec.criterion or "bic",
            fixed_params=fixed_params,
            preset=preset,
        )
    metric_fn = get_metric(metric)
    validation_splits, split_name, split_metadata, fold_ids = _resolve_selection_splits(
        frame.index,
        window=window,
        splits=splits,
        validation_splitter=spec.validation_splitter,
        allow_non_temporal_splits=allow_non_temporal_splits,
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
            fold_ids=fold_ids,
            score_aggregation=score_aggregation_value,
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
            fold_ids=fold_ids,
            score_aggregation=score_aggregation_value,
        )
    elif spec.method == "custom":
        rows, runtime_metadata = _run_custom_search(
            fit_model,
            frame,
            target,
            validation_splits,
            metric_fn,
            spec,
            base_params,
            rng,
            maximize=maximize,
            fold_ids=fold_ids,
            score_aggregation=score_aggregation_value,
        )
    else:
        candidates = _candidates(spec, rng)
        prefix_search = getattr(model_spec, "prefix_search", None) if model_spec is not None else None
        if prefix_search is not None and _all_candidates_have_key(candidates, prefix_search.param):
            rows = _evaluate_grid_with_prefix_groups(
                prefix_search,
                frame,
                target,
                validation_splits,
                metric_fn,
                base_params,
                candidates,
                fold_ids=fold_ids,
                score_aggregation=score_aggregation_value,
            )
        else:
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
                    fold_ids=fold_ids,
                    score_aggregation=score_aggregation_value,
                )
                for i, params in enumerate(candidates)
            ]

    trials = trial_frame(rows)
    ok = trials.loc[trials["status"] == "ok"].copy()
    if ok.empty:
        first_error = (
            trials["error"].dropna().iloc[0] if "error" in trials else "unknown error"
        )
        raise SearchError(f"All parameter trials failed: {first_error}", trials=trials)
    best_idx = ok["score"].idxmax() if maximize else ok["score"].idxmin()
    best_row = ok.loc[best_idx]
    successful_rows = {int(row.trial): row for row in rows if row.status == "ok"}
    best_trial = int(best_row["trial"])
    best_params = dict(successful_rows[best_trial].params)
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
            **_score_aggregation_metadata(score_aggregation_value),
            **runtime_metadata,
        },
    )


def _all_candidates_have_key(candidates: list[dict[str, Any]], key: str) -> bool:
    """Gate for the K-prefix grouped path: every candidate must search ``key``."""

    return bool(candidates) and all(key in candidate for candidate in candidates)


def _evaluate_grid_with_prefix_groups(
    prefix_search: PrefixSearchSpec,
    frame: pd.DataFrame,
    target: pd.Series,
    validation_splits: list[Split],
    metric_fn: Callable[[Any, Any], float],
    base_params: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    fold_ids: list[int] | None,
    score_aggregation: ScoreAggregation,
) -> list[SearchTrial]:
    """Group candidates identical except ``prefix_search.param`` and evaluate each
    group with a single shared fit per validation split (see ``evaluate_candidate_group``).
    Preserves the original ``enumerate(candidates)`` trial ids; grouping only changes
    which function computes each ``SearchTrial``, never the trial-id/candidate pairing.
    """

    groups: dict[tuple[Any, ...], list[tuple[int, dict[str, Any]]]] = {}
    group_order: list[tuple[Any, ...]] = []
    for trial_id, candidate_params in enumerate(candidates):
        trial_params = {**base_params, **candidate_params}
        key = _group_key(trial_params, prefix_search.param)
        if key not in groups:
            groups[key] = []
            group_order.append(key)
        groups[key].append((trial_id, candidate_params))
    rows: list[SearchTrial] = []
    for key in group_order:
        rows.extend(
            evaluate_candidate_group(
                prefix_search,
                frame,
                target,
                validation_splits,
                metric_fn,
                base_params,
                groups[key],
                fold_ids=fold_ids,
                score_aggregation=score_aggregation,
            )
        )
    return rows


def _group_key(trial_params: dict[str, Any], prefix_param: str) -> tuple[Any, ...]:
    """Canonical, hashable grouping key: every ``trial_params`` entry except the
    prefix param, sorted by key. Unhashable values fall back to ``repr()``."""

    items = sorted(
        (key, value) for key, value in trial_params.items() if key != prefix_param
    )
    canonical: list[tuple[Any, Any]] = []
    for key, value in items:
        try:
            hash(value)
            canonical.append((key, value))
        except TypeError:
            canonical.append((key, repr(value)))
    return tuple(canonical)


def _resolve_model(
    model: str | Callable[..., Any] | ModelSpec,
    *,
    preset: str | None,
) -> tuple[Callable[..., Any], ModelSpec | None]:
    if isinstance(model, (str, ModelSpec)):
        model_spec = _get_model_or_ensemble(model, preset=preset)
        return model_spec.fit, model_spec
    if callable(model):
        try:
            model_spec = _get_model_or_ensemble(model, preset=preset)
        except ValueError:
            if preset is not None:
                raise
            return model, None
        return model_spec.fit, model_spec
    raise TypeError("model must be a model name, callable, or ModelSpec")


def _get_model_or_ensemble(
    model: str | Callable[..., Any] | ModelSpec,
    *,
    preset: str | None = None,
) -> ModelSpec:
    try:
        return get_model(model, preset=preset)
    except ValueError as model_error:
        try:
            return get_model_ensemble(model, preset=preset)
        except ValueError:
            raise model_error from None


def _resolve_selection_xy(
    model_spec: ModelSpec | None,
    X: Any,
    y: Any | None,
    *,
    fixed_params: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    if model_spec is None:
        return resolve_xy(X, y)
    # Bare-series convenience: ``target``/``volatility`` models take a single series.
    # The autoregressive models (ar/far) are now ``supervised`` so the direct policy
    # can feed them lag features, but they still accept a bare series (the legacy
    # univariate AR usage / information-criterion order selection), in which case the
    # series IS the target and there are no separate features.
    autoregressive = "direct" in getattr(model_spec, "default_params", {})
    if (model_spec.input_kind in {"target", "volatility"} or autoregressive) and y is None:
        target = as_series(X)
        return pd.DataFrame(index=target.index), target
    if model_spec.input_kind == "panel":
        if y is None:
            frame = as_frame(X)
            target_column = _panel_target_column(
                model_spec, frame, fixed_params=fixed_params
            )
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

    def fit_panel_with_internal_target(
        X: Any, y: Any | None = None, **params: Any
    ) -> Any:
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
        "fixed_model_params": dict(model_spec.params),
        "backend": model_spec.backend,
        "requires_extra": model_spec.requires_extra,
        "requires_scaling": model_spec.requires_scaling,
        "recommended_preprocessing": model_spec.recommended_preprocessing,
    }


def _resolve_selection_splits(
    index: pd.Index,
    *,
    window: WindowSpec | str | None,
    splits: Sequence[tuple[Any, Any]] | None,
    validation_splitter: Any | None = None,
    allow_non_temporal_splits: bool = False,
) -> tuple[list[Split], str, dict[str, Any], list[int]]:
    n_obs = len(index)
    if splits is not None and window is not None:
        raise ValueError("pass either window or splits, not both")
    if splits is None:
        if validation_splitter is not None:
            resolved, split_name, metadata, fold_ids = _resolve_validation_splitter_with_fold_ids(
                index,
                validation_splitter,
            )
            temporal_order = bool(metadata.get("temporal_order", True))
            resolved = _normalize_splits(
                resolved,
                n_obs,
                allow_non_temporal_splits=allow_non_temporal_splits
                or not temporal_order,
            )
            return (
                resolved,
                split_name,
                {**metadata, "split_summary": _split_summary(resolved)},
                fold_ids,
            )
        window_spec = resolve_window(window)
        resolved = window_spec.split(n_obs)
        allow_non_temporal_splits = (
            allow_non_temporal_splits
            or window_spec.val.method == "random_kfold"
        )
        return (
            resolved,
            window_spec.to_dict()["method"],
            {
                "split_source": "window",
                "window": window_spec.to_dict(),
                "temporal_order": not allow_non_temporal_splits,
                "split_summary": _split_summary(resolved),
            },
            list(range(len(resolved))),
        )
    resolved = _normalize_splits(
        splits,
        n_obs,
        allow_non_temporal_splits=allow_non_temporal_splits,
    )
    return (
        resolved,
        "explicit_splits",
        {
            "split_source": "explicit",
            "window": None,
            "temporal_order": not allow_non_temporal_splits,
            "split_summary": _split_summary(resolved),
        },
        list(range(len(resolved))),
    )


def _normalize_splits(
    splits: Sequence[tuple[Any, Any]],
    n_obs: int,
    *,
    allow_non_temporal_splits: bool = False,
) -> list[Split]:
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
        if (
            not allow_non_temporal_splits
            and int(train_idx.max()) >= int(val_idx.min())
        ):
            raise ValueError(
                f"split {split_id} train positions must precede validation positions"
            )
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
        raise ValueError(
            f"split {split_id} {side} positions must not contain duplicates"
        )
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


def _resolve_score_aggregation(
    override: ScoreAggregation | None,
    spec_value: ScoreAggregation,
) -> ScoreAggregation:
    return _normalize_score_aggregation(
        spec_value if override is None else override
    )


def _score_aggregation_metadata(
    score_aggregation: ScoreAggregation,
) -> dict[str, str]:
    if score_aggregation == "mean_split":
        return {}
    return {"score_aggregation": score_aggregation}


def _prepare_search_spec(spec: SearchSpec) -> SearchSpec:
    prepared = replace(spec)
    prepared.method = _normalize_method(prepared.method)
    prepared.score_aggregation = _normalize_score_aggregation(
        prepared.score_aggregation
    )
    if prepared.criterion is not None:
        prepared.criterion = str(prepared.criterion).lower()
    prepared.param_grid = {
        key: _as_tuple(value) for key, value in prepared.param_grid.items()
    }
    prepared.param_distributions = {}
    for key, value in spec.param_distributions.items():
        try:
            prepared.param_distributions[key] = _coerce_distribution(value)
        except ValueError as exc:
            raise ValueError(f"invalid distribution for {key!r}: {exc}") from exc
    _validate_search_spec(prepared)
    return prepared


def _validate_search_spec(spec: SearchSpec) -> None:
    if spec.method not in {
        "fixed",
        "grid",
        "cv_path",
        "random",
        "bayesian",
        "genetic",
        "custom",
        "information_criterion",
    }:
        raise ValueError(f"Unknown search method {spec.method!r}")
    if spec.criterion is not None:
        criterion = str(spec.criterion).lower()
        if spec.method == "information_criterion" and criterion not in {"aic", "aicc", "bic"}:
            raise ValueError(
                "information_criterion search criterion must be 'aic', 'aicc', or 'bic'"
            )
    if spec.method == "information_criterion":
        if spec.param_distributions:
            raise ValueError("information_criterion search requires param_grid, not distributions")
        if spec.validation_splitter is not None:
            raise ValueError("information_criterion search does not use validation_splitter")
    if spec.n_iter < 1:
        raise ValueError("n_iter must be at least 1")
    if spec.population_size < 2:
        raise ValueError("population_size must be at least 2")
    if spec.generations < 1:
        raise ValueError("generations must be at least 1")
    if not 0 <= spec.mutation_rate <= 1:
        raise ValueError("mutation_rate must be between 0 and 1")
    if spec.method == "custom" and spec.custom_func is None:
        raise ValueError("custom search requires custom_func")
    if (
        spec.method in {"random", "bayesian", "genetic"}
        and not spec.param_distributions
    ):
        raise ValueError(
            f"{spec.method} search requires at least one parameter distribution"
        )
    for key, values in spec.param_grid.items():
        if not values:
            raise ValueError(f"parameter grid for {key!r} cannot be empty")
    for key, dist in spec.param_distributions.items():
        try:
            dist.validate()
        except ValueError as exc:
            raise ValueError(f"invalid distribution for {key!r}: {exc}") from exc


def _run_custom_search(
    model: Callable[..., Any],
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[Split],
    metric_fn: Callable[[Any, Any], float],
    spec: SearchSpec,
    fixed_params: dict[str, Any],
    rng: np.random.Generator,
    *,
    maximize: bool,
    fold_ids: list[int] | tuple[int, ...] | None = None,
    score_aggregation: ScoreAggregation = "mean_split",
) -> tuple[list[SearchTrial], dict[str, Any]]:
    if spec.custom_func is None:
        raise ValueError("custom search requires custom_func")
    resolved_fold_ids = tuple(fold_ids or range(len(splits)))
    default_score_aggregation = score_aggregation

    def evaluate_candidate_with_aggregation(
        candidate_model: Callable[..., Any],
        candidate_X: pd.DataFrame,
        candidate_y: pd.Series,
        candidate_splits: list[Split],
        candidate_metric: Callable[[Any, Any], float],
        candidate_fixed_params: dict[str, Any],
        candidate_params: dict[str, Any],
        candidate_trial: int,
        *,
        fold_ids: list[int] | tuple[int, ...] | None = None,
        score_aggregation: ScoreAggregation | None = None,
    ) -> SearchTrial:
        candidate_fold_ids = fold_ids
        if (
            candidate_fold_ids is None
            and len(candidate_splits) == len(resolved_fold_ids)
        ):
            candidate_fold_ids = resolved_fold_ids
        candidate_score_aggregation = (
            default_score_aggregation
            if score_aggregation is None
            else score_aggregation
        )
        return evaluate_candidate(
            candidate_model,
            candidate_X,
            candidate_y,
            candidate_splits,
            candidate_metric,
            candidate_fixed_params,
            candidate_params,
            candidate_trial,
            fold_ids=candidate_fold_ids,
            score_aggregation=candidate_score_aggregation,
        )

    output = spec.custom_func(
        model=model,
        X=X,
        y=y,
        splits=splits,
        metric=metric_fn,
        fixed_params=fixed_params,
        search=spec,
        rng=rng,
        maximize=maximize,
        evaluate_candidate=evaluate_candidate_with_aggregation,
        **spec.custom_params,
    )
    runtime_metadata: dict[str, Any] = {}
    if isinstance(output, tuple) and len(output) == 2:
        output, metadata = output
        if not isinstance(metadata, dict):
            raise TypeError("custom search metadata must be a dict")
        runtime_metadata.update(metadata)
    custom_metadata = dict(spec.metadata.get("custom_search", {}))
    custom_metadata.update(
        {
            "callable": _callable_name(spec.custom_func),
            "params": dict(spec.custom_params),
        }
    )
    runtime_metadata = {"custom_search": custom_metadata, **runtime_metadata}
    return _coerce_custom_trials(output), runtime_metadata


def _coerce_custom_trials(output: Any) -> list[SearchTrial]:
    if isinstance(output, SearchResult):
        output = output.trials
    if isinstance(output, pd.DataFrame):
        records = output.to_dict(orient="records")
        return [_trial_from_record(record, idx) for idx, record in enumerate(records)]
    if isinstance(output, SearchTrial):
        return [output]
    if isinstance(output, list) or isinstance(output, tuple):
        if all(isinstance(row, SearchTrial) for row in output):
            return list(output)
        if all(isinstance(row, dict) for row in output):
            return [_trial_from_record(record, idx) for idx, record in enumerate(output)]
    raise TypeError(
        "custom search callable must return SearchTrial records, a trial DataFrame, "
        "a SearchResult, or (records, metadata)"
    )


def _trial_from_record(record: dict[str, Any], fallback_trial: int) -> SearchTrial:
    reserved = {"trial", "score", "n_splits", "status", "error"}
    params = {key: value for key, value in record.items() if key not in reserved}
    score = record.get("score", np.nan)
    status = str(record.get("status", "ok"))
    return SearchTrial(
        trial=int(record.get("trial", fallback_trial)),
        params=params,
        score=float(score),
        n_splits=int(record.get("n_splits", 0)),
        status=status,
        error=None if pd.isna(record.get("error")) else str(record.get("error")),
    )


def _callable_name(func: Callable[..., Any]) -> str:
    module = getattr(func, "__module__", "")
    qualname = getattr(func, "__qualname__", getattr(func, "__name__", repr(func)))
    return f"{module}.{qualname}" if module else str(qualname)


__all__ = [
    "bayesian_search",
    "choice",
    "custom_search",
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


def _gaussian_information_criterion(
    ssr: float, nobs: int, n_params: int, criterion: str
) -> float:
    """Information criterion of a Gaussian OLS fit from its residual sum of squares.

    Uses the concentrated Gaussian log-likelihood, so AIC = 2k - 2 logL and
    BIC = k log n - 2 logL, with logL = -n/2 (log 2pi + log(SSR/n) + 1).
    """
    n = int(nobs)
    k = int(n_params)
    if n <= 0 or ssr <= 0.0:
        return float("inf")
    loglik = -0.5 * n * (math.log(2.0 * math.pi) + math.log(ssr / n) + 1.0)
    crit = str(criterion).lower()
    if crit == "aic":
        return 2.0 * k - 2.0 * loglik
    if crit == "aicc":
        denom = n - k - 1
        penalty = (2.0 * k * (k + 1) / denom) if denom > 0 else float("inf")
        return 2.0 * k - 2.0 * loglik + penalty
    if crit in {"bic", "sic", "schwarz"}:
        return k * math.log(n) - 2.0 * loglik
    raise ValueError("criterion must be one of: aic, aicc, bic")


def select_by_information_criterion(
    model: "str | Callable[..., Any] | ModelSpec",
    X: Any,
    y: Any | None = None,
    search: SearchSpec | None = None,
    *,
    criterion: str = "bic",
    fixed_params: dict[str, Any] | None = None,
    preset: str | None = None,
) -> SearchResult:
    """Select model hyperparameters by an in-sample information criterion.

    Unlike ``select_params``, each candidate is fitted on the whole supplied
    sample and scored by an information criterion (BIC by default) computed from
    the in-sample residual sum of squares and the parameter count, so no
    validation split is used. This matches the order selection the paper applies
    to the autoregression and the factor model. The fitted estimator must expose
    ``ssr_``, ``nobs_`` and ``n_params_``.
    """
    fit_model, model_spec = _resolve_model(model, preset=preset)
    frame, target = _resolve_selection_xy(model_spec, X, y, fixed_params=fixed_params)
    fit_model = _selection_fit_callable(fit_model, model_spec)
    spec = search or (
        _search_from_model(
            model_spec, method=None, random_state=None, n_iter=None,
            population_size=None, generations=None, mutation_rate=None,
        )
        if model_spec is not None
        else fixed()
    )
    spec = _prepare_search_spec(spec)
    rng = np.random.default_rng(spec.random_state)
    base_params = dict(fixed_params or {})
    candidates = _candidates(spec, rng)
    criterion_key = str(spec.criterion or criterion).lower()

    rows: list[dict[str, Any]] = []
    best: tuple[float, dict[str, Any]] | None = None
    for trial, candidate in enumerate(candidates):
        params = {**base_params, **candidate}
        try:
            fitted = fit_model(frame, target, **params)
            ssr, nobs, n_params = _ic_inputs(fitted, criterion=criterion_key)
            score = _gaussian_information_criterion(
                ssr,
                int(nobs),
                int(n_params),
                criterion_key,
            )
            rows.append({
                "trial": trial,
                **candidate,
                "score": float(score),
                "nobs": int(nobs),
                "n_params": int(n_params),
                "status": "ok",
                "error": None,
            })
        except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
            rows.append({
                "trial": trial,
                **candidate,
                "score": float("inf"),
                "status": "error",
                "error": str(exc),
            })
            continue
        if best is None or score < best[0]:
            best = (float(score), dict(candidate))

    trials = _ic_trial_frame(rows)
    if best is None:
        first_error = (
            trials["error"].dropna().iloc[0] if "error" in trials else "unknown error"
        )
        raise SearchError(
            f"All information-criterion parameter trials failed: {first_error}",
            trials=trials,
        )
    return SearchResult(
        best_params=dict(best[1]),
        best_score=float(best[0]),
        trials=trials,
        metric=criterion_key,
        method=f"information_criterion:{criterion_key}",
        window="none",
        metadata={
            "criterion": criterion_key,
            "n_candidates": int(len(candidates)),
            "validation": "none",
            **_model_metadata(model_spec),
        },
    )


def _ic_inputs(fitted: Any, *, criterion: str) -> tuple[float, int, int]:
    estimator = getattr(fitted, "estimator", fitted)
    ssr = getattr(estimator, "ssr_", None)
    nobs = getattr(estimator, "nobs_", None)
    n_params = getattr(estimator, "n_params_", None)
    if ssr is None or nobs is None or n_params is None:
        raise AttributeError(
            "information-criterion selection requires a fitted model exposing "
            f"ssr_, nobs_, and n_params_; {criterion!r} cannot be computed for "
            f"{type(estimator).__name__}"
        )
    return float(ssr), int(nobs), int(n_params)


def _ic_trial_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("information-criterion search produced no trials")
    for column in ("nobs", "n_params"):
        if column not in frame:
            frame[column] = np.nan
    first = ["trial"]
    last = ["score", "nobs", "n_params", "status", "error"]
    middle = [col for col in frame.columns if col not in set(first + last)]
    return frame[first + middle + last].sort_values("trial").reset_index(drop=True)
