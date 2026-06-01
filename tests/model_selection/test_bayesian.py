from __future__ import annotations

import pandas as pd
import pytest

import macroforecast as mf
from tests.model_selection.helpers import failing_model, first_prediction, score_model, xy


def test_bayesian_search_runs_gaussian_process_optimizer() -> None:
    X, y = xy()
    search = mf.model_selection.bayesian_search(
        {"score_value": [1.0, 3.0, 2.0]},
        n_iter=5,
        random_state=3,
    )

    result = mf.model_selection.select_params(
        score_model,
        X,
        y,
        search,
        window=mf.window.last_block(validation_size=6),
        metric=first_prediction,
        maximize=True,
    )

    assert result.method == "bayesian"
    assert result.metadata["optimizer"] == "gaussian_process_expected_improvement"
    assert result.metadata["initial_random_trials"] == 3
    assert result.metadata["candidate_pool_size"] >= 64
    assert "initial_random_trials" not in search.metadata
    assert "candidate_pool_size" not in search.metadata
    assert len(result.trials) == 5
    assert set(result.trials["status"]) == {"ok"}
    assert result.best_params == {"score_value": 3.0}


def test_bayesian_search_is_reproducible_with_mixed_distributions() -> None:
    X, y = xy()
    search = mf.model_selection.bayesian_search(
        {
            "score_value": mf.model_selection.uniform(0.0, 4.0),
            "max_leaf": mf.model_selection.randint(1, 3),
            "kind": ["a", "b"],
        },
        n_iter=6,
        random_state=17,
    )

    first = mf.model_selection.select_params(
        score_model,
        X,
        y,
        search,
        window=mf.window.last_block(validation_size=6),
        metric=first_prediction,
        maximize=True,
    )
    second = mf.model_selection.select_params(
        score_model,
        X,
        y,
        search,
        window=mf.window.last_block(validation_size=6),
        metric=first_prediction,
        maximize=True,
    )

    pd.testing.assert_frame_equal(first.trials, second.trials)
    assert first.best_params == second.best_params
    assert first.trials["score_value"].between(0.0, 4.0).all()
    assert set(first.trials["max_leaf"]).issubset({1, 2, 3})
    assert set(first.trials["kind"]).issubset({"a", "b"})
    assert "initial_random_trials" not in search.metadata


def test_bayesian_search_handles_small_finite_spaces() -> None:
    X, y = xy()
    search = mf.model_selection.bayesian_search(
        {"score_value": [1.0, 2.0]},
        n_iter=5,
        random_state=5,
    )

    result = mf.model_selection.select_params(
        score_model,
        X,
        y,
        search,
        window=mf.window.last_block(validation_size=6),
        metric=first_prediction,
        maximize=True,
    )

    assert len(result.trials) == 5
    assert set(result.trials["score_value"]).issubset({1.0, 2.0})
    assert set(result.trials["status"]) == {"ok"}


def test_bayesian_search_error_table_when_all_trials_fail() -> None:
    X, y = xy()
    search = mf.model_selection.bayesian_search(
        {"alpha": [0.1, 1.0]},
        n_iter=4,
        random_state=0,
    )

    with pytest.raises(mf.model_selection.SearchError) as excinfo:
        mf.model_selection.select_params(failing_model, X, y, search, window=mf.window.last_block(validation_size=6))

    assert len(excinfo.value.trials) == 4
    assert set(excinfo.value.trials["status"]) == {"error"}


def test_direct_bayesian_search_spec_gets_runtime_optimizer_metadata() -> None:
    X, y = xy()
    search = mf.model_selection.SearchSpec(
        method="bayesian",
        param_distributions={"score_value": [1.0, 2.0]},
        n_iter=3,
        random_state=2,
    )

    result = mf.model_selection.select_params(
        score_model,
        X,
        y,
        search,
        window=mf.window.last_block(validation_size=6),
        metric=first_prediction,
        maximize=True,
    )

    assert result.metadata["optimizer"] == "gaussian_process_expected_improvement"
    assert result.metadata["initial_random_trials"] == 3
    assert search.metadata == {}


def test_bayesian_search_falls_back_when_surrogate_fails(monkeypatch) -> None:
    from sklearn.gaussian_process import GaussianProcessRegressor

    X, y = xy()

    def fail_fit(self, X_train, y_train):  # noqa: ANN001
        raise RuntimeError("surrogate failed")

    monkeypatch.setattr(GaussianProcessRegressor, "fit", fail_fit)
    search = mf.model_selection.bayesian_search(
        {"score_value": [0.0, 1.0, 2.0, 3.0]},
        n_iter=5,
        random_state=9,
    )

    result = mf.model_selection.select_params(
        score_model,
        X,
        y,
        search,
        window=mf.window.last_block(validation_size=6),
        metric=first_prediction,
        maximize=True,
    )

    assert len(result.trials) == 5
    assert set(result.trials["status"]) == {"ok"}
    assert result.metadata["optimizer"] == "gaussian_process_expected_improvement"
