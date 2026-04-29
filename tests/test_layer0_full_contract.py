from __future__ import annotations

import pytest

from macrocast.compiler import CompileValidationError, compile_recipe_dict, compile_sweep_plan, run_compiled_recipe
from macrocast.defaults import build_default_recipe_dict


def _base_recipe(recipe_id: str = "layer0-base", *, model_families=("ar",), feature_builder: str = "target_lag_features") -> dict:
    return build_default_recipe_dict(
        dataset="fred_md",
        target="INDPRO",
        start="1980-01",
        end="1985-12",
        horizons=[1, 3],
        recipe_id=recipe_id,
        model_family=model_families[0],
        model_families=model_families,
        feature_builder=feature_builder,
    )


def _set_meta(recipe: dict, **values: str) -> None:
    recipe["path"]["0_meta"].setdefault("fixed_axes", {}).update(values)


def _set_leaf(recipe: dict, layer: str, **values: str) -> None:
    recipe["path"][layer].setdefault("leaf_config", {}).update(values)


def _fixed_to_sweep(recipe: dict, layer: str, axis: str, values: list[str]) -> None:
    fixed_axes = recipe["path"][layer].setdefault("fixed_axes", {})
    fixed_axes.pop(axis, None)
    recipe["path"][layer].setdefault("sweep_axes", {})[axis] = values


def _make_multi_target(recipe: dict) -> None:
    recipe["path"]["1_data_task"]["fixed_axes"]["target_structure"] = "multi_target"
    leaf = recipe["path"]["1_data_task"]["leaf_config"]
    leaf.pop("target", None)
    leaf["targets"] = ["INDPRO", "RPI"]


def test_single_default_is_direct_comparison_cell_executable() -> None:
    compiled = compile_recipe_dict(_base_recipe("single-default")).compiled

    assert compiled.execution_status == "executable"
    assert compiled.tree_context["route_owner"] == "comparison_sweep"
    assert compiled.tree_context["route_contract"] == "single_cell_executable"
    assert compiled.tree_context["study_scope"] == "one_target_one_method"


def test_model_comparison_parent_requires_sweep_runner_but_variants_are_executable() -> None:
    recipe = _base_recipe("model-comparison", model_families=("ar", "ridge"))

    parent = compile_recipe_dict(recipe).compiled
    assert parent.execution_status == "ready_for_sweep_runner"
    assert parent.tree_context["route_contract"] == "sweep_runner_executable"
    assert parent.tree_context["controlled_axis_kind"] == "model"

    plan = compile_sweep_plan(recipe)
    assert plan.size == 2
    variant_statuses = [compile_recipe_dict(variant.variant_recipe_dict).compiled.execution_status for variant in plan.variants]
    assert variant_statuses == ["executable", "executable"]


def test_feature_only_comparison_grid_is_sweep_runner_contract() -> None:
    recipe = _base_recipe("feature-comparison", model_families=("ridge",))
    _set_meta(recipe, study_scope="one_target_compare_methods")
    _fixed_to_sweep(recipe, "3_training", "feature_builder", ["target_lag_features", "raw_feature_panel"])

    compiled = compile_recipe_dict(recipe).compiled

    assert compiled.execution_status == "ready_for_sweep_runner"
    assert compiled.tree_context["route_owner"] == "comparison_sweep"
    assert compiled.tree_context["route_contract"] == "sweep_runner_executable"
    assert compiled.tree_context["controlled_axis_kind"] == "feature"
    assert compiled.tree_context["study_scope"] == "one_target_compare_methods"


def test_preprocessing_sweep_is_runner_ready_but_variants_can_be_blocked_by_layer2() -> None:
    recipe = _base_recipe("preprocessing-sweep", model_families=("ridge",))
    _set_meta(recipe, study_scope="one_target_compare_methods")
    _fixed_to_sweep(recipe, "2_preprocessing", "scaling_policy", ["none", "standard"])

    compiled = compile_recipe_dict(recipe).compiled
    assert compiled.execution_status == "ready_for_sweep_runner"
    assert compiled.tree_context["route_contract"] == "sweep_runner_executable"
    assert compiled.tree_context["controlled_axis_kind"] == "preprocessing"

    plan = compile_sweep_plan(recipe)
    statuses: list[str] = []
    for variant in plan.variants:
        try:
            statuses.append(compile_recipe_dict(variant.variant_recipe_dict).compiled.execution_status)
        except CompileValidationError:
            statuses.append("compile_error")
    assert statuses == ["executable", "compile_error"]


@pytest.mark.parametrize("unit", ["single_target_full_sweep", "benchmark_suite", "ablation_study", "replication_recipe"])
def test_removed_study_scope_values_are_rejected(unit: str) -> None:
    recipe = _base_recipe(f"removed-{unit}", model_families=("ridge",))
    _set_meta(recipe, study_scope=unit)

    with pytest.raises(CompileValidationError, match="unknown value"):
        compile_recipe_dict(recipe)


def test_multiple_targets_compare_methods_is_comparison_sweep_ready() -> None:
    recipe = _base_recipe("multi-target-compare", model_families=("ar", "ridge"))
    _make_multi_target(recipe)
    _set_meta(recipe, study_scope="multiple_targets_compare_methods")

    compiled = compile_recipe_dict(recipe).compiled

    assert compiled.run_spec.route_owner == "comparison_sweep"
    assert compiled.execution_status == "ready_for_sweep_runner"
    assert compiled.tree_context["route_contract"] == "sweep_runner_executable"
    assert compiled.tree_context["study_scope"] == "multiple_targets_compare_methods"



def test_multiple_targets_one_method_stays_direct_comparison_cell_executable() -> None:
    recipe = _base_recipe("multi-target-shared")
    _make_multi_target(recipe)
    _set_meta(recipe, study_scope="multiple_targets_one_method")

    compiled = compile_recipe_dict(recipe).compiled

    assert compiled.execution_status == "executable"
    assert compiled.run_spec.route_owner == "comparison_sweep"
    assert compiled.tree_context["route_contract"] == "single_cell_executable"
    assert compiled.tree_context["study_scope"] == "multiple_targets_one_method"


def test_route_contract_is_preserved_in_manifest() -> None:
    recipe = _base_recipe("manifest-contract", model_families=("ar", "ridge"))

    manifest = compile_recipe_dict(recipe).manifest

    assert manifest["tree_context"]["route_contract"] == "sweep_runner_executable"
    assert manifest["tree_context"]["variation_axes"] == ["3_training.model_family"]
