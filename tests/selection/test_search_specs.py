from __future__ import annotations

import pytest

import macroforecast as mf
from tests.selection.helpers import xy


def test_search_spec_uses_model_owned_space_without_fitting() -> None:
    search = mf.selection.search_spec(
        "random_forest",
        preset="small",
        method="random",
        n_iter=5,
        random_state=42,
    )

    assert search.method == "random"
    assert search.n_iter == 5
    assert set(search.param_distributions) == {
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
    }
    assert search.metadata["model"] == "random_forest"
    assert (
        search.to_dict()["param_distributions"]["n_estimators"]["kind"] == "categorical"
    )


def test_search_spec_threads_genetic_options_from_model_space() -> None:
    search = mf.selection.search_spec(
        "decision_tree",
        preset="small",
        method="genetic",
        population_size=4,
        generations=2,
        mutation_rate=0.35,
        random_state=7,
    )

    assert search.method == "genetic"
    assert search.population_size == 4
    assert search.generations == 2
    assert search.mutation_rate == 0.35
    assert set(search.param_distributions) == {"max_depth", "min_samples_leaf"}
    assert search.metadata["model"] == "decision_tree"


def test_search_spec_includes_model_preprocessing_metadata() -> None:
    search = mf.selection.search_spec("svr", preset="small", method="grid")

    assert search.metadata["model"] == "svr"
    assert search.metadata["backend"] == "sklearn.svm.SVR"
    assert search.metadata["requires_scaling"] is True
    assert search.metadata["recommended_preprocessing"] == (
        "standardize predictors before fitting",
    )


def test_search_spec_and_result_export_json_ready(tmp_path) -> None:
    X, y = xy()
    search = mf.selection.search_spec(
        "ridge",
        preset="small",
    )

    search_path = tmp_path / "search.json"
    search_json = search.to_json(search_path)
    search_dict = search.to_dict()

    assert '"model": "ridge"' in search_json
    assert search_path.read_text(encoding="utf-8").startswith("{")
    assert search.to_metadata()["metadata"]["model_preset"] == "small"
    assert search_dict["param_grid"]["alpha"] == [0.01, 0.1, 1.0]

    result = mf.selection.select_params(
        "ridge", X, y, search, window=mf.window.last_block(validation_size=8)
    )
    result_path = tmp_path / "result.json"
    result_json = result.to_json(result_path)
    result_dict = result.to_dict()

    assert result.to_metadata()["n_trials"] == 3
    assert result_dict["metadata"]["model"] == "ridge"
    assert len(result_dict["trials"]) == 3
    assert "trials" not in result.to_dict(include_trials=False)
    assert '"trials"' not in result.to_json(include_trials=False)
    assert '"trials"' in result_json
    assert result_path.read_text(encoding="utf-8").startswith("{")


def test_explicit_search_spec_is_normalized_and_coerced() -> None:
    X, y = xy()
    search = mf.selection.SearchSpec(
        method="RANDOM_SEARCH",
        param_distributions={"alpha": [0.01, 0.1]},
        n_iter=2,
        random_state=0,
    )

    result = mf.selection.select_params(
        mf.models.ridge,
        X,
        y,
        search,
        window=mf.window.last_block(validation_size=6),
    )

    assert result.method == "random"
    assert len(result.trials) == 2
    assert set(result.trials["status"]) == {"ok"}


def test_support_vector_selection_runs_with_model_owned_space() -> None:
    X, y = xy()

    result = mf.selection.select_params(
        "svr",
        X,
        y,
        search=mf.selection.grid({"C": [0.1, 1.0], "epsilon": [0.01]}),
        window=mf.window.last_block(validation_size=6),
    )

    assert set(result.best_params) == {"C", "epsilon"}
    assert result.metadata["model"] == "svr"
    assert result.metadata["requires_scaling"] is True
    assert set(result.trials["status"]) == {"ok"}


def test_nn_selection_runs_with_fixed_training_params() -> None:
    pytest.importorskip("torch")
    X, y = xy()

    result = mf.selection.select_params(
        "nn",
        X,
        y,
        search=mf.selection.grid(
            {"hidden_layer_sizes": [(4,)], "weight_decay": [0.0, 0.0001]}
        ),
        window=mf.window.last_block(validation_size=6),
        fixed_params={
            "max_epochs": 1,
            "batch_size": 8,
            "random_state": 0,
            "device": "cpu",
        },
    )

    assert result.best_params["hidden_layer_sizes"] == (4,)
    assert result.best_params["max_epochs"] == 1
    assert result.best_params["device"] == "cpu"
    assert result.metadata["model"] == "nn"
    assert result.metadata["backend"] == "torch.nn.Sequential"
    assert result.metadata["requires_extra"] == "deep"


def test_invalid_explicit_search_spec_errors_early() -> None:
    X, y = xy()
    search = mf.selection.SearchSpec(method="random", param_distributions={}, n_iter=2)

    with pytest.raises(
        ValueError, match="requires at least one parameter distribution"
    ):
        mf.selection.select_params(mf.models.ridge, X, y, search)


def test_search_json_export_falls_back_for_custom_metadata() -> None:
    search = mf.selection.fixed({"alpha": 0.1})
    search.metadata["callable"] = lambda x: x

    exported = search.to_dict()

    assert "function" in exported["metadata"]["callable"]


def test_distribution_builders_validate_bounds_early() -> None:
    with pytest.raises(ValueError, match="low < high"):
        mf.selection.uniform(1.0, 1.0)
    with pytest.raises(ValueError, match="positive bounds"):
        mf.selection.log_uniform(0.0, 1.0)
    with pytest.raises(ValueError, match="low <= high"):
        mf.selection.randint(5, 4)
    with pytest.raises(ValueError, match="at least one choice"):
        mf.selection.choice([])


def test_explicit_search_spec_validates_distribution_bounds() -> None:
    X, y = xy()
    search = mf.selection.SearchSpec(
        method="random",
        param_distributions={
            "alpha": mf.selection.ParamDistribution("log_float", low=-1.0, high=1.0)
        },
        n_iter=2,
    )

    with pytest.raises(ValueError, match="invalid distribution for 'alpha'"):
        mf.selection.select_params(
            mf.models.ridge,
            X,
            y,
            search,
            window=mf.window.last_block(validation_size=6),
        )


def test_public_top_level_selection_exports() -> None:
    assert mf.select_params is mf.selection.select_params
    assert mf.search_spec is mf.selection.search_spec
    assert not hasattr(mf, "make_search")
    assert not hasattr(mf, "SearchTrial")
    assert not hasattr(mf.selection, "SearchTrial")
    assert not hasattr(mf.selection, "split_table")
    assert not hasattr(mf.selection, "rmse")
    assert mf.SearchSpec is mf.selection.SearchSpec
    assert mf.SearchError is mf.selection.SearchError
    assert mf.rmse is mf.evaluation.rmse
    assert mf.split_table is mf.window.split_table
    assert mf.normalize_window_name("holdout") == "last_block"
    assert "selection" in dir(mf)
    assert "evaluation" in dir(mf)
    assert "window" in dir(mf)
