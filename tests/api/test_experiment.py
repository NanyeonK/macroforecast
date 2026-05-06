"""Tests for the v0.8.0 ``mf.Experiment`` builder class."""
from __future__ import annotations

import pytest
import yaml as _yaml

import macroforecast as mf
from macroforecast.api_high import ForecastResult, _set_at
from macroforecast.core.execution import execute_recipe

from ._offline import install_custom_panel


# ---------------------------------------------------------------------------
# Construction + run smoke
# ---------------------------------------------------------------------------

def test_experiment_run_returns_forecast_result(tmp_path):
    exp = mf.Experiment(
        dataset="fred_md",
        target="y",
        horizons=[1],
        random_seed=0,
        model_family="ridge",
    )
    install_custom_panel(exp)
    result = exp.run(output_directory=tmp_path)
    assert isinstance(result, ForecastResult)
    assert len(result.cells) == 1
    assert result.cells[0].succeeded
    assert result.manifest_path is not None
    assert result.manifest_path.exists()


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------

def test_experiment_compare_models_expands_to_two_cells(tmp_path):
    exp = mf.Experiment(
        dataset="fred_md",
        target="y",
        horizons=[1],
        random_seed=0,
        model_family="ridge",
    )
    install_custom_panel(exp)
    exp.compare_models(["ridge", "ols"])
    result = exp.run(output_directory=tmp_path)
    assert len(result.cells) == 2
    assert all(cell.succeeded for cell in result.cells)
    # Each cell records the swept family in its sweep_values dict.
    swept_families = sorted(
        v for cell in result.cells for v in cell.sweep_values.values()
    )
    assert swept_families == ["ols", "ridge"]


def test_experiment_compare_models_rejects_empty_list():
    exp = mf.Experiment(dataset="fred_md", target="y", horizons=[1])
    with pytest.raises(ValueError, match="at least one"):
        exp.compare_models([])


# ---------------------------------------------------------------------------
# compare() / sweep()
# ---------------------------------------------------------------------------

def test_experiment_compare_generic_axis(tmp_path):
    exp = mf.Experiment(
        dataset="fred_md",
        target="y",
        horizons=[1],
        random_seed=0,
        model_family="ridge",
    )
    install_custom_panel(exp)
    # The L4 fit node id is "fit_1_ridge" by RecipeBuilder convention.
    exp.compare(
        "4_forecasting_model.nodes.fit_1_ridge.params.alpha",
        [0.1, 1.0],
    )
    result = exp.run(output_directory=tmp_path)
    assert len(result.cells) == 2
    assert all(cell.succeeded for cell in result.cells)


def test_experiment_sweep_is_alias_for_compare():
    exp1 = mf.Experiment(dataset="fred_md", target="y", horizons=[1], model_family="ridge")
    exp2 = mf.Experiment(dataset="fred_md", target="y", horizons=[1], model_family="ridge")
    exp1.compare("4_forecasting_model.nodes.fit_1_ridge.params.alpha", [0.1, 1.0])
    exp2.sweep("4_forecasting_model.nodes.fit_1_ridge.params.alpha", [0.1, 1.0])
    assert exp1.to_recipe_dict() == exp2.to_recipe_dict()


def test_experiment_compare_rejects_empty_values():
    exp = mf.Experiment(dataset="fred_md", target="y", horizons=[1], model_family="ridge")
    with pytest.raises(ValueError, match="at least one"):
        exp.compare("4_forecasting_model.nodes.fit_1_ridge.params.alpha", [])


def test_experiment_compare_rejects_empty_path():
    exp = mf.Experiment(dataset="fred_md", target="y", horizons=[1])
    with pytest.raises(ValueError, match="non-empty"):
        exp.compare("", [1, 2])


def test_experiment_compare_rejects_unknown_list_id():
    exp = mf.Experiment(dataset="fred_md", target="y", horizons=[1], model_family="ridge")
    with pytest.raises(ValueError, match="no list entry with id"):
        exp.compare(
            "4_forecasting_model.nodes.does_not_exist.params.alpha",
            [0.1, 1.0],
        )


# ---------------------------------------------------------------------------
# variant() -- v0.8.5 implementation
# ---------------------------------------------------------------------------

def test_experiment_variant_records_overrides():
    exp = mf.Experiment(dataset="fred_md", target="y", horizons=[1])
    exp.variant("alt", model="ols")
    recipe = exp.to_recipe_dict()
    assert recipe["variants"] == {"alt": {"model_family": "ols"}}


def test_experiment_variant_rejects_invalid_name():
    exp = mf.Experiment(dataset="fred_md", target="y", horizons=[1])
    with pytest.raises(ValueError, match="variant name"):
        exp.variant("bad=name")


def test_experiment_variant_chain_returns_self():
    exp = mf.Experiment(dataset="fred_md", target="y", horizons=[1])
    result = exp.variant("a", model="ridge").variant("b", model="lasso")
    assert result is exp
    assert set(exp.to_recipe_dict()["variants"]) == {"a", "b"}


# ---------------------------------------------------------------------------
# to_yaml / to_recipe_dict round-trip
# ---------------------------------------------------------------------------

def test_experiment_to_yaml_round_trips_through_execute_recipe(tmp_path):
    exp = mf.Experiment(
        dataset="fred_md",
        target="y",
        horizons=[1],
        random_seed=0,
        model_family="ridge",
    )
    install_custom_panel(exp)
    yaml_text = exp.to_yaml()
    recipe = _yaml.safe_load(yaml_text)
    manifest = execute_recipe(recipe, output_directory=tmp_path)
    assert manifest.cells and manifest.cells[0].succeeded


def test_experiment_to_recipe_dict_returns_deep_copy():
    exp = mf.Experiment(dataset="fred_md", target="y", horizons=[1], model_family="ridge")
    snapshot = exp.to_recipe_dict()
    snapshot["0_meta"]["leaf_config"]["random_seed"] = 9999
    again = exp.to_recipe_dict()
    assert again["0_meta"]["leaf_config"]["random_seed"] != 9999


# ---------------------------------------------------------------------------
# replicate
# ---------------------------------------------------------------------------

def test_experiment_replicate_matches_sink_hashes(tmp_path):
    exp = mf.Experiment(
        dataset="fred_md",
        target="y",
        horizons=[1],
        random_seed=0,
        model_family="ridge",
    )
    install_custom_panel(exp)
    result = exp.run(output_directory=tmp_path)
    manifest_path = result.manifest_path
    assert manifest_path is not None
    replication = exp.replicate(manifest_path)
    assert replication.sink_hashes_match


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def test_experiment_validate_passes_for_well_formed_offline_recipe():
    exp = mf.Experiment(
        dataset="fred_md",
        target="y",
        horizons=[1],
        random_seed=0,
        model_family="ridge",
    )
    install_custom_panel(exp)
    # Should not raise.
    exp.validate()


# ---------------------------------------------------------------------------
# _set_at edge cases
# ---------------------------------------------------------------------------

def test_set_at_creates_intermediate_dicts():
    root: dict = {}
    _set_at(root, "a.b.c", 7)
    assert root == {"a": {"b": {"c": 7}}}


def test_set_at_overwrites_existing_sweep_marker():
    root: dict = {"a": {"b": {"sweep": [1, 2]}}}
    _set_at(root, "a.b", {"sweep": [3, 4, 5]})
    assert root == {"a": {"b": {"sweep": [3, 4, 5]}}}


def test_set_at_walks_into_list_by_id():
    root = {"nodes": [{"id": "n1", "params": {"alpha": 1.0}}]}
    _set_at(root, "nodes.n1.params.alpha", 2.0)
    assert root["nodes"][0]["params"]["alpha"] == 2.0


def test_set_at_rejects_traversal_into_scalar():
    root: dict = {"a": 1}
    # ``a`` is a leaf int -- both the descent and the final-step branch
    # detect the non-collection and refuse with a clear message.
    with pytest.raises(ValueError):
        _set_at(root, "a.b", 2)
