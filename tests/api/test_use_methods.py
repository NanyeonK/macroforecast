"""Tests for the v0.8.5 ``Experiment.use_*`` hooks + variant() execution wiring."""
from __future__ import annotations

import pytest

import macroforecast as mf

from ._offline import install_custom_panel


# ---------------------------------------------------------------------------
# .use_fred_sd_selection
# ---------------------------------------------------------------------------

def test_use_fred_sd_selection_emits_l1_axes_and_leafs():
    exp = mf.Experiment(dataset="fred_md+fred_sd", target="INDPRO", horizons=[1])
    exp.use_fred_sd_selection(states=["CA", "TX"], variables=["UR"])
    recipe = exp.to_recipe_dict()
    fixed = recipe["1_data"]["fixed_axes"]
    leaf = recipe["1_data"]["leaf_config"]
    assert fixed["state_selection"] == "selected_states"
    assert fixed["sd_variable_selection"] == "selected_sd_variables"
    assert leaf["selected_states"] == ["CA", "TX"]
    assert leaf["selected_sd_variables"] == ["UR"]


def test_use_fred_sd_selection_states_only():
    exp = mf.Experiment(dataset="fred_md+fred_sd", target="INDPRO", horizons=[1])
    exp.use_fred_sd_selection(states=["NY"])
    recipe = exp.to_recipe_dict()
    fixed = recipe["1_data"]["fixed_axes"]
    leaf = recipe["1_data"]["leaf_config"]
    assert fixed["state_selection"] == "selected_states"
    assert "sd_variable_selection" not in fixed
    assert leaf["selected_states"] == ["NY"]


def test_use_fred_sd_selection_returns_self():
    exp = mf.Experiment(dataset="fred_md+fred_sd", target="INDPRO", horizons=[1])
    assert exp.use_fred_sd_selection(states=["CA"]) is exp


# ---------------------------------------------------------------------------
# .use_fred_sd_state_group / .use_fred_sd_variable_group
# ---------------------------------------------------------------------------

def test_use_fred_sd_state_group_sets_axis():
    exp = mf.Experiment(dataset="fred_md+fred_sd", target="INDPRO", horizons=[1])
    exp.use_fred_sd_state_group("census_region_west")
    recipe = exp.to_recipe_dict()
    assert recipe["1_data"]["fixed_axes"]["fred_sd_state_group"] == "census_region_west"


def test_use_fred_sd_state_group_rejects_unknown():
    exp = mf.Experiment(dataset="fred_md+fred_sd", target="INDPRO", horizons=[1])
    with pytest.raises(ValueError, match="not a known option"):
        exp.use_fred_sd_state_group("nonexistent")


def test_use_fred_sd_variable_group_sets_axis():
    exp = mf.Experiment(dataset="fred_md+fred_sd", target="INDPRO", horizons=[1])
    exp.use_fred_sd_variable_group("housing")
    recipe = exp.to_recipe_dict()
    assert recipe["1_data"]["fixed_axes"]["fred_sd_variable_group"] == "housing"


def test_use_fred_sd_variable_group_rejects_unknown():
    exp = mf.Experiment(dataset="fred_md+fred_sd", target="INDPRO", horizons=[1])
    with pytest.raises(ValueError, match="not a known option"):
        exp.use_fred_sd_variable_group("does_not_exist")


# ---------------------------------------------------------------------------
# .use_mixed_frequency_representation
# ---------------------------------------------------------------------------

def test_use_mixed_frequency_representation_sets_axis():
    exp = mf.Experiment(dataset="fred_md+fred_sd", target="INDPRO", horizons=[1])
    exp.use_mixed_frequency_representation("native_frequency_block_payload")
    recipe = exp.to_recipe_dict()
    assert (
        recipe["2_preprocessing"]["fixed_axes"]["mixed_frequency_representation"]
        == "native_frequency_block_payload"
    )


def test_use_mixed_frequency_representation_rejects_unknown():
    exp = mf.Experiment(dataset="fred_md+fred_sd", target="INDPRO", horizons=[1])
    with pytest.raises(ValueError, match="not valid"):
        exp.use_mixed_frequency_representation("aligned")


# ---------------------------------------------------------------------------
# .use_sd_inferred_tcodes
# ---------------------------------------------------------------------------

def test_use_sd_inferred_tcodes_sets_l2_axis():
    exp = mf.Experiment(dataset="fred_md+fred_sd", target="INDPRO", horizons=[1])
    exp.use_sd_inferred_tcodes()
    recipe = exp.to_recipe_dict()
    assert recipe["2_preprocessing"]["fixed_axes"]["sd_tcode_policy"] == "inferred"


# ---------------------------------------------------------------------------
# .use_sd_empirical_tcodes
# ---------------------------------------------------------------------------

def test_use_sd_empirical_tcodes_variable_global():
    exp = mf.Experiment(dataset="fred_md+fred_sd", target="INDPRO", horizons=[1])
    exp.use_sd_empirical_tcodes("variable_global", audit_uri="audit.csv")
    recipe = exp.to_recipe_dict()
    fixed = recipe["2_preprocessing"]["fixed_axes"]
    leaf = recipe["2_preprocessing"]["leaf_config"]
    assert fixed["sd_tcode_policy"] == "empirical"
    assert leaf["sd_tcode_unit"] == "variable_global"
    assert leaf["sd_tcode_audit_uri"] == "audit.csv"
    assert "sd_tcode_code_map" not in leaf


def test_use_sd_empirical_tcodes_state_series():
    exp = mf.Experiment(dataset="fred_md+fred_sd", target="INDPRO", horizons=[1])
    exp.use_sd_empirical_tcodes(
        "state_series",
        code_map={"UR_CA": 2, "UR_TX": 5},
        audit_uri="state_audit.csv",
    )
    recipe = exp.to_recipe_dict()
    leaf = recipe["2_preprocessing"]["leaf_config"]
    assert leaf["sd_tcode_unit"] == "state_series"
    assert leaf["sd_tcode_code_map"] == {"UR_CA": 2, "UR_TX": 5}
    assert leaf["sd_tcode_audit_uri"] == "state_audit.csv"


def test_use_sd_empirical_tcodes_state_series_requires_code_map():
    exp = mf.Experiment(dataset="fred_md+fred_sd", target="INDPRO", horizons=[1])
    with pytest.raises(ValueError, match="code_map"):
        exp.use_sd_empirical_tcodes("state_series")


def test_use_sd_empirical_tcodes_rejects_unknown_unit():
    exp = mf.Experiment(dataset="fred_md+fred_sd", target="INDPRO", horizons=[1])
    with pytest.raises(ValueError, match="not valid"):
        exp.use_sd_empirical_tcodes("series_global")


# ---------------------------------------------------------------------------
# .use_preprocessor
# ---------------------------------------------------------------------------

def test_use_preprocessor_sets_l2_leaf():
    exp = mf.Experiment(dataset="fred_md", target="y", horizons=[1])
    exp.use_preprocessor("custom_x_demean")
    recipe = exp.to_recipe_dict()
    assert recipe["2_preprocessing"]["leaf_config"]["custom_postprocessor"] == "custom_x_demean"


def test_use_preprocessor_l2_raises_not_implemented():
    exp = mf.Experiment(dataset="fred_md", target="y", horizons=[1])
    with pytest.raises(NotImplementedError, match="v0.9"):
        exp.use_preprocessor("foo", applied_at="l2")


# ---------------------------------------------------------------------------
# Chain test
# ---------------------------------------------------------------------------

def test_chain_use_methods_round_trips_recipe():
    exp = (
        mf.Experiment(dataset="fred_md+fred_sd", target="INDPRO", horizons=[1])
        .use_fred_sd_selection(states=["CA"], variables=["UR"])
        .use_mixed_frequency_representation("calendar_aligned_frame")
        .compare_models(["ridge", "ols"])
    )
    recipe = exp.to_recipe_dict()
    assert recipe["1_data"]["fixed_axes"]["state_selection"] == "selected_states"
    assert (
        recipe["2_preprocessing"]["fixed_axes"]["mixed_frequency_representation"]
        == "calendar_aligned_frame"
    )
    # compare_models should still have inserted a sweep marker on family.
    fit_node = next(
        n for n in recipe["4_forecasting_model"]["nodes"]
        if isinstance(n, dict) and n.get("op") == "fit_model"
    )
    assert fit_node["params"]["family"] == {"sweep": ["ridge", "ols"]}


# ---------------------------------------------------------------------------
# variant() runtime expansion
# ---------------------------------------------------------------------------

def test_variant_alone_expands_to_one_cell_per_variant(tmp_path):
    exp = mf.Experiment(
        dataset="fred_md", target="y", horizons=[1], model_family="ridge",
    )
    install_custom_panel(exp)
    exp.variant("ridge_default").variant("ols_alt", model="ols")
    result = exp.run(output_directory=tmp_path)
    assert len(result.cells) == 2
    assert all(cell.succeeded for cell in result.cells)
    cell_ids = {cell.cell_id for cell in result.cells}
    # Both variant names should appear in the cell_id strings.
    assert any("ridge_default" in c for c in cell_ids)
    assert any("ols_alt" in c for c in cell_ids)


def test_variant_plus_compare_models_cross_product(tmp_path):
    exp = mf.Experiment(
        dataset="fred_md", target="y", horizons=[1], model_family="ridge",
    )
    install_custom_panel(exp)
    # variant overrides one knob, compare_models sweeps another.
    exp.variant("v1", **{"0_meta.leaf_config.random_seed": 0})
    exp.variant("v2", **{"0_meta.leaf_config.random_seed": 1})
    exp.compare_models(["ridge", "ols"])
    result = exp.run(output_directory=tmp_path)
    # 2 variants × 2 models = 4 cells.
    assert len(result.cells) == 4
    assert all(cell.succeeded for cell in result.cells)


def test_variant_plus_sweep_grid(tmp_path):
    exp = mf.Experiment(
        dataset="fred_md", target="y", horizons=[1], model_family="ridge",
    )
    install_custom_panel(exp)
    exp.variant("v1", **{"0_meta.leaf_config.random_seed": 0})
    exp.variant("v2", **{"0_meta.leaf_config.random_seed": 1})
    exp.sweep("4_forecasting_model.nodes.fit_1_ridge.params.alpha", [0.1, 1.0])
    result = exp.run(output_directory=tmp_path)
    # 2 variants × 2 alpha = 4 cells.
    assert len(result.cells) == 4
    assert all(cell.succeeded for cell in result.cells)
