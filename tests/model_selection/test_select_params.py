from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from tests.model_selection.helpers import (
    failing_model,
    first_prediction,
    panel_no_leak_fit,
    score_model,
    xy,
)


def test_grid_select_params_returns_trial_table() -> None:
    X, y = xy()
    search = mf.model_selection.grid(
        {"alpha": [0.001, 0.1, 10.0]}
    )

    result = mf.model_selection.select_params(
        mf.models.ridge,
        X,
        y,
        search,
        window=mf.window.last_block(validation_size=8),
    )

    assert isinstance(result, mf.model_selection.SearchResult)
    assert result.best_params["alpha"] in {0.001, 0.1, 10.0}
    assert len(result.trials) == 3
    assert set(result.trials["status"]) == {"ok"}


def test_select_params_can_use_model_owned_space_from_model_spec() -> None:
    X, y = xy()
    model = mf.models.get_model("ridge", preset="small")

    result = mf.model_selection.select_params(
        model,
        X,
        y,
        window=mf.window.last_block(validation_size=8),
    )

    assert result.method == "cv_path"
    assert set(result.trials["alpha"]) == {0.01, 0.1, 1.0}
    assert result.metadata["model"] == "ridge"
    assert result.metadata["model_preset"] == "small"


def test_select_params_can_use_model_name_and_search_method_override() -> None:
    X, y = xy()

    result = mf.model_selection.select_params(
        "decision_tree",
        X,
        y,
        preset="small",
        method="grid",
        window=mf.window.last_block(validation_size=8),
    )

    assert result.method == "grid"
    assert {"max_depth", "min_samples_leaf"}.issubset(result.trials.columns)
    assert result.metadata["model"] == "decision_tree"


def test_select_params_can_use_model_ensemble_spec_with_fixed_params() -> None:
    X, y = xy()
    spec = mf.model_ensemble.get_model_ensemble(
        "bagging",
        params={"base": "ridge", "n_estimators": 2},
    )

    result = mf.model_selection.select_params(
        spec,
        X,
        y,
        search=mf.model_selection.grid({"max_samples": [0.5, 0.8]}),
        window=mf.window.last_block(validation_size=6),
    )

    assert result.metadata["model"] == "bagging"
    assert result.metadata["model_family"] == "model_ensemble"
    assert result.metadata["fixed_model_params"] == {"base": "ridge", "n_estimators": 2}
    assert set(result.best_params) == {"max_samples"}
    assert set(result.trials["status"]) == {"ok"}


def test_select_params_can_use_panel_model_with_separate_target() -> None:
    X, y = xy()

    result = mf.model_selection.select_params(
        "var",
        X,
        y,
        preset="small",
        method="grid",
        window=mf.window.last_block(validation_size=6),
    )

    assert result.metadata["model"] == "var"
    assert set(result.trials["n_lag"]) == {1, 2, 4}


def test_select_params_drops_named_panel_target_when_y_is_omitted() -> None:
    X, y = xy()
    panel = pd.concat([X["x1"].rename("noise"), y.rename("actual"), X["x2"]], axis=1)
    model = mf.models.ModelSpec(
        name="panel_no_leak",
        family="test",
        fit_func=panel_no_leak_fit,
        default_params={"target": "actual"},
        search_spaces={"standard": {"k": (1,)}},
        input_kind="panel",
    )

    result = mf.model_selection.select_params(
        model,
        panel,
        window=mf.window.last_block(validation_size=6),
    )

    assert result.best_params == {"k": 1}
    assert set(result.trials["status"]) == {"ok"}


def test_select_params_can_use_target_only_model_without_X() -> None:
    _, y = xy()

    result = mf.model_selection.select_params(
        "ar",
        y,
        preset="small",
        method="grid",
        window=mf.window.last_block(validation_size=6),
    )

    assert result.metadata["model"] == "ar"
    assert set(result.trials["n_lag"]) == {1, 2, 4}


def test_select_params_threads_genetic_options() -> None:
    X, y = xy()

    result = mf.model_selection.select_params(
        "decision_tree",
        X,
        y,
        preset="small",
        method="genetic",
        population_size=4,
        generations=2,
        mutation_rate=0.4,
        window=mf.window.last_block(validation_size=8),
        random_state=11,
    )

    assert result.method == "genetic"
    assert len(result.trials) == 8
    assert set(result.trials["status"]) == {"ok"}


def test_all_failed_trials_raise_search_error_with_trial_table() -> None:
    X, y = xy()
    search = mf.model_selection.grid(
        {"alpha": [0.1, 1.0]}
    )

    with pytest.raises(mf.model_selection.SearchError) as excinfo:
        mf.model_selection.select_params(failing_model, X, y, search, window=mf.window.last_block(validation_size=6))

    trials = excinfo.value.trials
    assert len(trials) == 2
    assert set(trials["status"]) == {"error"}
    assert "intentional failure" in trials["error"].iloc[0]


def test_select_params_stores_canonical_window_name() -> None:
    X, y = xy()
    search = mf.model_selection.grid({"alpha": [0.1, 1.0]})

    result = mf.model_selection.select_params(mf.models.ridge, X, y, search, window="holdout")

    assert result.window == "last_block"


def test_select_params_supports_custom_metric_and_maximize() -> None:
    X, y = xy()
    search = mf.model_selection.grid(
        {"score_value": [1.0, 3.0, 2.0]}
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

    assert result.best_params == {"score_value": 3.0}
    assert result.best_score == 3.0
    assert result.trials["score"].max() == 3.0


def test_select_params_aligns_array_predictions_to_validation_index() -> None:
    X, y = xy()

    class ArrayFit:
        def __init__(self, value: float) -> None:
            self.value = float(value)

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.full(len(X), self.value)

    def array_model(X: pd.DataFrame, y: pd.Series, *, value: float = 0.0) -> ArrayFit:
        return ArrayFit(value)

    result = mf.model_selection.select_params(
        array_model,
        X,
        y,
        mf.model_selection.grid({"value": [0.0, 1.0]}),
        window=mf.window.last_block(validation_size=6),
    )

    assert set(result.trials["status"]) == {"ok"}
    assert result.best_params == {"value": 1.0}


def test_random_search_is_reproducible() -> None:
    X, y = xy()
    search = mf.model_selection.random_search(
        {
            "alpha": mf.model_selection.log_uniform(0.001, 1.0),
            "max_iter": [1000, 2000],
        },
        n_iter=4,
        random_state=123,
    )

    first = mf.model_selection.select_params(
        mf.models.lasso,
        X,
        y,
        search,
        window=mf.window.last_block(validation_size=6),
    )
    second = mf.model_selection.select_params(
        mf.models.lasso,
        X,
        y,
        search,
        window=mf.window.last_block(validation_size=6),
    )

    pd.testing.assert_frame_equal(first.trials, second.trials)
    assert first.best_params == second.best_params


def test_cv_path_is_one_parameter_grid() -> None:
    X, y = xy()
    search = mf.model_selection.cv_path(
        param="alpha",
        values=[0.01, 0.1],
    )

    result = mf.model_selection.select_params(mf.models.ridge, X, y, search, window=mf.window.poos(validation_size=3))

    assert result.method == "cv_path"
    assert set(result.trials["alpha"]) == {0.01, 0.1}
    assert result.metadata["path_param"] == "alpha"


def test_fixed_search_handles_none_parameter_values() -> None:
    X, y = xy()
    search = mf.model_selection.fixed(
        {"max_depth": None, "min_samples_leaf": 2}
    )

    result = mf.model_selection.select_params(
        mf.models.decision_tree,
        X,
        y,
        search,
        window=mf.window.last_block(validation_size=5),
    )

    assert result.best_params["max_depth"] is None
    assert result.best_params["min_samples_leaf"] == 2


def test_explicit_search_rejects_separate_search_overrides() -> None:
    X, y = xy()
    search = mf.model_selection.grid({"alpha": [0.1, 1.0]})

    with pytest.raises(ValueError, match="search was provided"):
        mf.model_selection.select_params("ridge", X, y, search, preset="small", population_size=4)


def test_select_params_accepts_explicit_integer_splits() -> None:
    X, y = xy()
    search = mf.model_selection.grid({"alpha": [0.1, 1.0]})
    splits = [
        (np.arange(0, 24), np.arange(24, 30)),
        (np.arange(0, 30), np.arange(30, 36)),
    ]

    result = mf.model_selection.select_params(mf.models.ridge, X, y, search, splits=splits)

    assert result.window == "explicit_splits"
    assert result.metadata["split_source"] == "explicit"
    assert result.metadata["window"] is None
    assert result.metadata["n_splits"] == 2
    assert result.metadata["split_summary"][1]["validation_end_pos"] == 35
    assert set(result.trials["n_splits"]) == {2}


def test_select_params_rejects_ambiguous_or_invalid_splits() -> None:
    X, y = xy()
    search = mf.model_selection.grid({"alpha": [0.1]})

    with pytest.raises(ValueError, match="either window or splits"):
        mf.model_selection.select_params(
            mf.models.ridge,
            X,
            y,
            search,
            window=mf.window.last_block(validation_size=6),
            splits=[(np.arange(0, 24), np.arange(24, 30))],
        )
    with pytest.raises(ValueError, match="overlap"):
        mf.model_selection.select_params(
            mf.models.ridge,
            X,
            y,
            search,
            splits=[(np.arange(0, 24), np.arange(20, 30))],
        )
    with pytest.raises(ValueError, match="must precede validation"):
        mf.model_selection.select_params(
            mf.models.ridge,
            X,
            y,
            search,
            splits=[(np.arange(12, 24), np.arange(6, 12))],
        )
    with pytest.raises(TypeError, match="integer positions"):
        mf.model_selection.select_params(
            mf.models.ridge,
            X,
            y,
            search,
            splits=[(np.arange(0, 24), X.index[24:30])],
        )
