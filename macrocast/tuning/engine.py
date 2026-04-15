from __future__ import annotations

from .budget import TuningBudget
from .hp_spaces import MODEL_HP_SPACES
from .search.grid import grid_search
from .search.random import random_search
from .search.bayesian import bayesian_optimization
from .search.genetic import genetic_algorithm
from .types import TuningResult, TuningSpec
from .validation.scorer import get_scorer
from .validation.splitter import LastBlockSplitter, RollingBlocksSplitter, ExpandingValidationSplitter, BlockedKFoldSplitter
import numpy as np


def resolve_validation_size(rule: str, total_train_size: int, config: dict) -> int:
    if rule == "ratio":
        return max(1, int(total_train_size * float(config.get("ratio", 0.2))))
    if rule == "fixed_n":
        return min(int(config.get("n", 5)), max(1, total_train_size // 2))
    if rule == "fixed_years":
        obs_per_year = int(config.get("obs_per_year", 12))
        return min(int(config.get("years", 1)) * obs_per_year, max(1, total_train_size // 2))
    raise ValueError(f"unknown validation_size_rule: {rule}")


def build_splitter(spec: TuningSpec, n_samples: int):
    val_size = resolve_validation_size(spec.validation_size_rule, n_samples, spec.validation_size_config)
    gap = int(spec.embargo_gap_size)
    if spec.validation_location == "last_block":
        return LastBlockSplitter(val_size, gap)
    if spec.validation_location == "rolling_blocks":
        return RollingBlocksSplitter(n_blocks=3, block_size=val_size, embargo_gap=gap)
    if spec.validation_location == "expanding_validation":
        return ExpandingValidationSplitter(max(2, n_samples - val_size), step_size=1, embargo_gap=gap)
    if spec.validation_location == "blocked_cv":
        return BlockedKFoldSplitter(n_splits=min(5, max(2, n_samples // max(1, val_size))), embargo_gap=gap)
    raise ValueError(f"unknown validation_location: {spec.validation_location}")


def _evaluate_hp(model_factory, hp: dict, X_train: np.ndarray, y_train: np.ndarray, splitter, scorer) -> float:
    scores=[]
    for train_idx, val_idx in splitter.split(len(X_train)):
        if len(train_idx) < 2 or len(val_idx) < 1:
            continue
        model = model_factory(hp)
        model.fit(X_train[train_idx], y_train[train_idx])
        pred = model.predict(X_train[val_idx])
        scores.append(float(scorer(y_train[val_idx], pred)))
    if not scores:
        return float("inf")
    return float(np.mean(scores))


def _expand_discrete_grid(hp_space: dict):
    grid={}
    for name, dist in hp_space.items():
        if dist.type == "categorical":
            grid[name]=list(dist.choices)
        elif dist.type == "int":
            vals=[int(dist.low), int((int(dist.low)+int(dist.high))//2), int(dist.high)]
            grid[name]=sorted(set(vals))
        elif dist.type in {"float","log_float"}:
            low=float(dist.low); high=float(dist.high)
            if dist.type == "log_float":
                vals=[low, float(np.sqrt(low*high)), high]
            else:
                vals=[low, (low+high)/2.0, high]
            grid[name]=vals
    return grid


def run_tuning(model_family: str, model_factory, X_train: np.ndarray, y_train: np.ndarray, tuning_spec: TuningSpec) -> TuningResult:
    hp_space = tuning_spec.hp_space or MODEL_HP_SPACES.get(model_family, {})
    scorer = get_scorer(tuning_spec.tuning_objective)
    splitter = build_splitter(tuning_spec, len(X_train))
    budget = TuningBudget(
        max_trials=tuning_spec.tuning_budget.get("max_trials"),
        max_time_seconds=tuning_spec.tuning_budget.get("max_time_seconds"),
        early_stop_trials=tuning_spec.tuning_budget.get("early_stop_trials"),
        min_improvement=float(tuning_spec.tuning_budget.get("min_improvement", 0.0)),
    )
    eval_fn = lambda hp: _evaluate_hp(model_factory, hp, X_train, y_train, splitter, scorer)
    algo=tuning_spec.search_algorithm
    if algo == "grid_search":
        return grid_search(eval_fn, _expand_discrete_grid(hp_space), budget)
    if algo == "random_search":
        return random_search(eval_fn, hp_space, budget, random_state=tuning_spec.seed or 42)
    if algo == "bayesian_optimization":
        return bayesian_optimization(eval_fn, hp_space, budget, random_state=tuning_spec.seed or 42)
    if algo == "genetic_algorithm":
        return genetic_algorithm(eval_fn, hp_space, budget, random_state=tuning_spec.seed or 42)
    raise ValueError(f"unknown search_algorithm: {algo}")
