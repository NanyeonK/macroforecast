from __future__ import annotations

from collections.abc import Callable
from typing import Any
import warnings

import numpy as np
import pandas as pd

from macroforecast.selection.runner import evaluate_candidate
from macroforecast.selection.types import ParamDistribution, SearchSpec, SearchTrial
from macroforecast.window import Split


def sample_params(
    distributions: dict[str, ParamDistribution],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Sample one parameter dictionary from named parameter distributions."""

    return {key: dist.sample(rng) for key, dist in distributions.items()}


def run_genetic(
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
) -> list[SearchTrial]:
    """Run lightweight genetic search over parameter distributions."""

    population = [sample_params(spec.param_distributions, rng) for _ in range(spec.population_size)]
    rows: list[SearchTrial] = []
    trial = 0
    for _ in range(spec.generations):
        generation_rows = []
        for params in population:
            row = evaluate_candidate(model, X, y, splits, metric_fn, fixed_params, params, trial)
            trial += 1
            rows.append(row)
            if row.status == "ok":
                generation_rows.append((row, params))
        if not generation_rows:
            population = [
                sample_params(spec.param_distributions, rng)
                for _ in range(spec.population_size)
            ]
            continue
        generation_rows.sort(key=lambda item: item[0].score, reverse=maximize)
        survivors = [params for _, params in generation_rows[: max(1, len(generation_rows) // 2)]]
        population = []
        while len(population) < spec.population_size:
            parent_a = survivors[int(rng.integers(0, len(survivors)))]
            parent_b = survivors[int(rng.integers(0, len(survivors)))]
            child = {}
            for key, dist in spec.param_distributions.items():
                value = parent_a[key] if rng.random() < 0.5 else parent_b[key]
                if rng.random() < spec.mutation_rate:
                    value = dist.sample(rng)
                child[key] = value
            population.append(child)
    return rows


def run_bayesian(
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
) -> tuple[list[SearchTrial], dict[str, Any]]:
    """Run a sampled-pool Gaussian-process expected-improvement optimizer."""

    initial = min(spec.n_iter, max(3, len(spec.param_distributions) + 1))
    pool_size = max(64, 32 * len(spec.param_distributions), 8 * spec.n_iter)
    metadata = {
        "optimizer": spec.metadata.get("optimizer", "gaussian_process_expected_improvement"),
        "initial_random_trials": initial,
        "candidate_pool_size": pool_size,
    }
    rows: list[SearchTrial] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for trial in range(spec.n_iter):
        if trial < initial:
            params = sample_unique_params(spec.param_distributions, rng, seen)
        else:
            params = propose_bayesian_candidate(
                spec.param_distributions,
                rows,
                rng,
                seen,
                pool_size=pool_size,
                maximize=maximize,
            )
        seen.add(candidate_key(params))
        rows.append(evaluate_candidate(model, X, y, splits, metric_fn, fixed_params, params, trial))
    return rows, metadata


def propose_bayesian_candidate(
    distributions: dict[str, ParamDistribution],
    rows: list[SearchTrial],
    rng: np.random.Generator,
    seen: set[tuple[tuple[str, str], ...]],
    *,
    pool_size: int,
    maximize: bool,
) -> dict[str, Any]:
    """Propose the next candidate from a sampled pool using expected improvement."""

    successful = [row for row in rows if row.status == "ok" and np.isfinite(row.score)]
    if len(successful) < 2:
        return sample_unique_params(distributions, rng, seen)

    try:
        from scipy.stats import norm
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    except Exception:  # pragma: no cover - dependencies are declared required.
        return sample_unique_params(distributions, rng, seen)

    X_train = np.vstack([encode_params(row.params, distributions) for row in successful])
    scores = np.asarray([row.score for row in successful], dtype=float)
    objective = -scores if maximize else scores
    pool = candidate_pool(distributions, rng, seen, pool_size=pool_size)
    if not pool:
        return sample_unique_params(distributions, rng, seen)
    X_pool = np.vstack([encode_params(params, distributions) for params in pool])
    kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-6)
    surrogate = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            surrogate.fit(X_train, objective)
            mean, std = surrogate.predict(X_pool, return_std=True)
    except Exception:
        return sample_unique_params(distributions, rng, seen)

    std = np.maximum(std, 1e-12)
    improvement = np.min(objective) - mean
    z = improvement / std
    expected_improvement = improvement * norm.cdf(z) + std * norm.pdf(z)
    return pool[int(np.argmax(expected_improvement))]


def candidate_pool(
    distributions: dict[str, ParamDistribution],
    rng: np.random.Generator,
    seen: set[tuple[tuple[str, str], ...]],
    *,
    pool_size: int,
) -> list[dict[str, Any]]:
    """Sample a finite candidate pool excluding known parameter combinations."""

    pool: list[dict[str, Any]] = []
    pool_seen = set(seen)
    for _ in range(max(pool_size * 4, 100)):
        params = sample_params(distributions, rng)
        key = candidate_key(params)
        if key in pool_seen:
            continue
        pool.append(params)
        pool_seen.add(key)
        if len(pool) >= pool_size:
            break
    return pool


def sample_unique_params(
    distributions: dict[str, ParamDistribution],
    rng: np.random.Generator,
    seen: set[tuple[tuple[str, str], ...]],
) -> dict[str, Any]:
    """Sample parameters while attempting to avoid previously seen combinations."""

    params = sample_params(distributions, rng)
    for _ in range(100):
        key = candidate_key(params)
        if key not in seen:
            return params
        params = sample_params(distributions, rng)
    return params


def candidate_key(params: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Return a hashable representation for duplicate candidate detection."""

    return tuple((key, repr(value)) for key, value in sorted(params.items()))


def encode_params(
    params: dict[str, Any],
    distributions: dict[str, ParamDistribution],
) -> np.ndarray:
    """Encode mixed parameter dictionaries into a numeric surrogate-model row."""

    return np.asarray(
        [encode_value(params[key], dist) for key, dist in distributions.items()],
        dtype=float,
    )


def encode_value(value: Any, dist: ParamDistribution) -> float:
    """Encode one parameter value according to its distribution type."""

    if dist.kind == "float":
        assert dist.low is not None and dist.high is not None
        return scale(float(value), float(dist.low), float(dist.high))
    if dist.kind == "log_float":
        assert dist.low is not None and dist.high is not None
        return scale(np.log(float(value)), np.log(float(dist.low)), np.log(float(dist.high)))
    if dist.kind == "int":
        assert dist.low is not None and dist.high is not None
        return scale(float(value), float(dist.low), float(dist.high))
    if dist.kind == "categorical":
        try:
            index = dist.choices.index(value)
        except ValueError:
            index = 0
        if len(dist.choices) <= 1:
            return 0.0
        return float(index) / float(len(dist.choices) - 1)
    raise ValueError(f"Unknown distribution kind: {dist.kind!r}")


def scale(value: float, low: float, high: float) -> float:
    """Scale a scalar into a unit interval coordinate."""

    if high == low:
        return 0.0
    return float((value - low) / (high - low))


__all__ = [
    "candidate_key",
    "candidate_pool",
    "encode_params",
    "encode_value",
    "propose_bayesian_candidate",
    "run_bayesian",
    "run_genetic",
    "sample_params",
    "sample_unique_params",
    "scale",
]
