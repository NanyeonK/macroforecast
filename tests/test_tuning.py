from __future__ import annotations

import numpy as np

from macrocast.execution.deep_training import build_factor_panel, fit_with_optional_tuning
from macrocast.tuning import HPDistribution, TuningBudget, run_tuning
from macrocast.tuning.types import TuningSpec
from macrocast.tuning.validation.splitter import LastBlockSplitter, RollingBlocksSplitter, ExpandingValidationSplitter, BlockedKFoldSplitter
from macrocast.tuning.validation.scorer import get_scorer


class DummyModel:
    def __init__(self, hp=None):
        self.bias = float((hp or {}).get("bias", 0.0))
    def fit(self, X, y):
        self.mean_ = float(np.mean(y)) + self.bias
        return self
    def predict(self, X):
        return np.full(len(X), self.mean_)


def test_hp_distribution_sampling() -> None:
    rng = np.random.RandomState(42)
    assert 0.0 <= HPDistribution("float", 0.0, 1.0).sample(rng) <= 1.0
    assert 1 <= HPDistribution("int", 1, 3).sample(rng) <= 3
    assert HPDistribution("categorical", choices=("a", "b")).sample(rng) in {"a", "b"}


def test_temporal_splitters_produce_nonempty_splits() -> None:
    n = 20
    splitters = [LastBlockSplitter(4), RollingBlocksSplitter(3, 4), ExpandingValidationSplitter(5), BlockedKFoldSplitter(4)]
    for splitter in splitters:
        splits = list(splitter.split(n))
        assert splits
        for train_idx, val_idx in splits:
            assert len(train_idx) > 0 and len(val_idx) > 0


def test_get_scorer_validation_mse() -> None:
    scorer = get_scorer("validation_mse")
    assert scorer(np.array([1.0, 2.0]), np.array([1.0, 3.0])) == 0.5


def test_run_tuning_grid_random_bayes_genetic() -> None:
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.linspace(0.0, 1.0, 20)
    hp_space = {"bias": HPDistribution("float", -0.1, 0.1)}
    for algo in ["grid_search", "random_search", "bayesian_optimization", "genetic_algorithm"]:
        spec = TuningSpec(
            search_algorithm=algo,
            tuning_objective="validation_mse",
            tuning_budget={"max_trials": 4, "max_time_seconds": 5.0, "early_stop_trials": 2},
            hp_space=hp_space,
            validation_size_rule="ratio",
            validation_size_config={"ratio": 0.2, "n": 3, "years": 1, "obs_per_year": 12},
            validation_location="last_block",
            embargo_gap="none",
            embargo_gap_size=0,
            seed=42,
        )
        result = run_tuning("ridge", lambda hp: DummyModel(hp), X, y, spec)
        assert result.total_trials >= 1
        assert isinstance(result.best_hp, dict)


def test_tuning_budget_loss_plateau_uses_min_improvement() -> None:
    budget = TuningBudget(early_stop_trials=2, min_improvement=0.05)
    budget.update(1.00)
    assert budget._no_improvement_count == 0
    budget.update(0.98)
    assert budget._no_improvement_count == 1
    budget.update(0.97)
    assert budget._no_improvement_count == 2
    assert budget.exceeded() is True


def test_fit_with_optional_tuning_supports_adaptive_lasso_and_boosting() -> None:
    X = np.arange(80, dtype=float).reshape(40, 2)
    y = np.linspace(0.0, 1.0, 40)
    training_spec = {
        "enable_tuning": True,
        "search_algorithm": "random_search",
        "tuning_objective": "validation_mse",
        "validation_size_rule": "ratio",
        "validation_ratio": 0.2,
        "validation_location": "last_block",
        "max_trials": 3,
        "max_time_seconds": 5.0,
        "early_stopping": "validation_patience",
        "early_stop_trials": 2,
        "random_seed": 42,
    }
    for model_family in ["adaptive_lasso", "componentwise_boosting", "boosting_ridge", "boosting_lasso"]:
        model, payload = fit_with_optional_tuning(model_family, X, y, training_spec)
        pred = model.predict(X[:2])
        assert pred.shape == (2,)
        assert payload["total_trials"] >= 1
        assert isinstance(payload["best_hp"], dict)


def test_build_factor_panel_respects_target_lag_count_gt_one() -> None:
    X_train = __import__("pandas").DataFrame(np.arange(60, dtype=float).reshape(12, 5))
    X_pred = __import__("pandas").DataFrame(np.arange(5, dtype=float).reshape(1, 5))
    y = np.arange(12, dtype=float)
    X_aug, X_pred_aug = build_factor_panel(
        X_train,
        y,
        X_pred,
        {"fixed_factor_count": 2, "factor_count": "fixed", "target_lag_count": 2},
        include_ar_lags=True,
    )
    assert X_aug.shape[0] == len(y) - 2
    assert X_aug.shape[1] == 4
    assert X_pred_aug.shape == (1, 4)
