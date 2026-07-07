from __future__ import annotations

from collections.abc import Callable, Iterable
from itertools import product
from typing import Any

import numpy as np

from macroforecast.model_ensemble import get_model_ensemble
from macroforecast.models.specs import ModelSpec, get_model
from macroforecast.model_selection.optimizers import sample_params
from macroforecast.model_selection.types import (
    ParamDistribution,
    SearchSpec,
    ValidationSplitterSpec,
)


def uniform(low: float, high: float) -> ParamDistribution:
    """Continuous uniform distribution."""

    return _validated_distribution(ParamDistribution("float", low=low, high=high))


def log_uniform(low: float, high: float) -> ParamDistribution:
    """Continuous log-uniform distribution for positive parameters."""

    return _validated_distribution(ParamDistribution("log_float", low=low, high=high))


def randint(low: int, high: int) -> ParamDistribution:
    """Inclusive integer distribution."""

    return _validated_distribution(ParamDistribution("int", low=low, high=high))


def choice(values: Iterable[Any]) -> ParamDistribution:
    """Categorical distribution over explicit values."""

    return _validated_distribution(
        ParamDistribution("categorical", choices=tuple(values))
    )


def fixed(
    params: dict[str, Any] | None = None,
    *,
    random_state: int | None = None,
) -> SearchSpec:
    """Evaluate one fixed parameter set without tuning."""

    return SearchSpec(
        method="fixed",
        param_grid={key: (value,) for key, value in (params or {}).items()},
        random_state=random_state,
    )


def search_spec(
    model: str | Callable[..., Any] | ModelSpec,
    *,
    preset: str | None = None,
    method: str | None = None,
    random_state: int | None = None,
    n_iter: int | None = None,
    population_size: int | None = None,
    generations: int | None = None,
    mutation_rate: float | None = None,
) -> SearchSpec:
    """Build a SearchSpec from a registered model's owned search space."""

    model_spec = _get_model_or_ensemble(model, preset=preset)
    return _search_from_model(
        model_spec,
        method=method,
        random_state=random_state,
        n_iter=n_iter,
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
    )


def grid(
    param_grid: dict[str, Iterable[Any] | Any],
    *,
    validation_splitter: ValidationSplitterSpec | Callable[..., Any] | str | None = None,
) -> SearchSpec:
    """Grid-search over explicit parameter values."""

    return SearchSpec(
        method="grid",
        param_grid={key: _as_tuple(value) for key, value in param_grid.items()},
        validation_splitter=validation_splitter,
    )


def random_search(
    param_distributions: dict[str, ParamDistribution | Iterable[Any] | Any],
    *,
    n_iter: int = 20,
    random_state: int | None = None,
) -> SearchSpec:
    """Seeded random search over parameter distributions."""

    return SearchSpec(
        method="random",
        param_distributions={
            key: _coerce_distribution(value)
            for key, value in param_distributions.items()
        },
        n_iter=n_iter,
        random_state=random_state,
    )


def cv_path(
    *,
    param: str = "alpha",
    values: Iterable[Any] | None = None,
) -> SearchSpec:
    """Evaluate an ordered one-parameter path, commonly lasso/ridge alpha values."""

    path_values = (
        tuple(values) if values is not None else (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
    )
    spec = grid({param: path_values})
    spec.method = "cv_path"
    spec.metadata["path_param"] = param
    return spec


def bayesian_search(
    param_distributions: dict[str, ParamDistribution | Iterable[Any] | Any],
    *,
    n_iter: int = 20,
    random_state: int | None = None,
) -> SearchSpec:
    """Sequential Gaussian-process Bayesian search request."""

    spec = random_search(param_distributions, n_iter=n_iter, random_state=random_state)
    spec.method = "bayesian"
    spec.metadata["optimizer"] = "gaussian_process_expected_improvement"
    return spec


def genetic_search(
    param_distributions: dict[str, ParamDistribution | Iterable[Any] | Any],
    *,
    population_size: int = 12,
    generations: int = 4,
    mutation_rate: float = 0.2,
    random_state: int | None = None,
) -> SearchSpec:
    """Lightweight genetic-style stochastic search over parameter distributions."""

    if population_size < 2:
        raise ValueError("population_size must be at least 2")
    if generations < 1:
        raise ValueError("generations must be at least 1")
    if not 0 <= mutation_rate <= 1:
        raise ValueError("mutation_rate must be between 0 and 1")
    return SearchSpec(
        method="genetic",
        param_distributions={
            key: _coerce_distribution(value)
            for key, value in param_distributions.items()
        },
        random_state=random_state,
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
    )


def custom_search(
    name: str,
    func: Callable[..., Any],
    *,
    param_grid: dict[str, Iterable[Any] | Any] | None = None,
    param_distributions: dict[str, ParamDistribution | Iterable[Any] | Any] | None = None,
    n_iter: int = 20,
    random_state: int | None = None,
    metadata: dict[str, Any] | None = None,
    **params: Any,
) -> SearchSpec:
    """Build a user-supplied parameter-search request."""

    if not name:
        raise ValueError("custom search name must be non-empty")
    if not callable(func):
        raise TypeError("custom search func must be callable")
    custom_metadata = {
        "custom_search": {
            "name": str(name),
            "callable": _callable_name(func),
            "params": dict(params),
        },
        **dict(metadata or {}),
    }
    return SearchSpec(
        method="custom",
        param_grid={key: _as_tuple(value) for key, value in (param_grid or {}).items()},
        param_distributions={
            key: _coerce_distribution(value)
            for key, value in (param_distributions or {}).items()
        },
        n_iter=n_iter,
        random_state=random_state,
        custom_func=func,
        custom_params=dict(params),
        metadata=custom_metadata,
    )


def _search_from_model(
    model_spec: ModelSpec | None,
    *,
    method: str | None,
    random_state: int | None,
    n_iter: int | None,
    population_size: int | None,
    generations: int | None,
    mutation_rate: float | None,
) -> SearchSpec:
    if model_spec is None:
        return fixed(random_state=random_state)
    space = model_spec.search_space()
    search_space: dict[str, Any] = dict(space)
    method_name = _normalize_method(method or model_spec.default_search_method)
    if not search_space:
        spec = fixed(random_state=random_state)
    elif method_name == "cv_path":
        if len(search_space) != 1:
            raise ValueError(
                "cv_path requires exactly one tunable parameter in the model search space"
            )
        param, values = next(iter(search_space.items()))
        spec = cv_path(param=param, values=values)
    elif method_name == "grid":
        spec = grid(search_space)
    elif method_name == "random":
        spec = random_search(
            search_space,
            n_iter=20 if n_iter is None else n_iter,
            random_state=random_state,
        )
    elif method_name == "bayesian":
        spec = bayesian_search(
            search_space,
            n_iter=20 if n_iter is None else n_iter,
            random_state=random_state,
        )
    elif method_name == "genetic":
        spec = genetic_search(
            search_space,
            population_size=12 if population_size is None else population_size,
            generations=4 if generations is None else generations,
            mutation_rate=0.2 if mutation_rate is None else mutation_rate,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model search method {method_name!r}")
    spec.metadata.update(
        {
            "model": model_spec.name,
            "model_family": model_spec.family,
            "model_preset": model_spec.preset,
            "backend": model_spec.backend,
            "requires_extra": model_spec.requires_extra,
            "requires_scaling": model_spec.requires_scaling,
            "recommended_preprocessing": model_spec.recommended_preprocessing,
        }
    )
    return spec


def _normalize_method(method: str) -> str:
    key = str(method).lower().replace("-", "_")
    aliases = {
        "grid_search": "grid",
        "random_search": "random",
        "bayesian_search": "bayesian",
        "genetic_search": "genetic",
        "custom_search": "custom",
        "cvpath": "cv_path",
        "path": "cv_path",
        "ic": "information_criterion",
        "informationcriterion": "information_criterion",
    }
    return aliases.get(key, key)


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


def _as_tuple(value: Iterable[Any] | Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        values = value
    elif isinstance(value, list):
        values = tuple(value)
    elif isinstance(value, range):
        values = tuple(value)
    elif isinstance(value, np.ndarray):
        values = tuple(value.tolist())
    else:
        values = (value,)
    if not values:
        raise ValueError("parameter grids cannot contain empty value lists")
    return values


def _coerce_distribution(
    value: ParamDistribution | Iterable[Any] | Any,
) -> ParamDistribution:
    if isinstance(value, ParamDistribution):
        value.validate()
        return value
    return choice(_as_tuple(value))


def _validated_distribution(distribution: ParamDistribution) -> ParamDistribution:
    distribution.validate()
    return distribution


def _callable_name(func: Callable[..., Any]) -> str:
    module = getattr(func, "__module__", "")
    qualname = getattr(func, "__qualname__", getattr(func, "__name__", repr(func)))
    return f"{module}.{qualname}" if module else str(qualname)


def _candidates(spec: SearchSpec, rng: np.random.Generator) -> list[dict[str, Any]]:
    if spec.method in {"fixed", "grid", "cv_path", "information_criterion"}:
        if not spec.param_grid:
            return [{}]
        keys = list(spec.param_grid)
        return [
            dict(zip(keys, values, strict=True))
            for values in product(*(spec.param_grid[k] for k in keys))
        ]
    if spec.method == "random":
        if spec.n_iter < 1:
            raise ValueError("n_iter must be at least 1")
        return [
            sample_params(spec.param_distributions, rng) for _ in range(spec.n_iter)
        ]
    raise ValueError(f"Unknown search method {spec.method!r}")


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
    "search_spec",
    "uniform",
]
