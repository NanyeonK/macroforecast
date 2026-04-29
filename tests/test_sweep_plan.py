"""Tests for macrocast.compiler.sweep_plan (Phase 1 sub-task 01.1 / 01.7)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from macrocast.compiler import compile_recipe_dict
from macrocast.compiler.sweep_plan import (
    DEFAULT_MAX_VARIANTS,
    SweepPlan,
    SweepPlanError,
    SweepVariant,
    compile_sweep_plan,
)


def _base_recipe() -> dict:
    return {
        "recipe_id": "sweep-test",
        "path": {
            "0_meta": {"fixed_axes": {"study_scope": "one_target_compare_methods"}},
            "1_data_task": {
                "fixed_axes": {"dataset": "fred_md", "target_structure": "single_target"},
                "leaf_config": {"target": "INDPRO", "horizons": [1]},
            },
            "2_preprocessing": {"fixed_axes": {"scaling_policy": "standard"}},
            "3_training": {
                "fixed_axes": {"framework": "expanding", "benchmark_family": "autoregressive_bic"},
                "sweep_axes": {"model_family": ["ridge", "lasso", "elasticnet"]},
            },
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
        },
    }


def test_single_axis_sweep_expands_to_cartesian_variants() -> None:
    plan = compile_sweep_plan(_base_recipe())

    assert isinstance(plan, SweepPlan)
    assert plan.size == 3
    assert plan.parent_recipe_id == "sweep-test"
    assert plan.axes_swept == ("3_training.model_family",)
    values = {v.axis_values["3_training.model_family"] for v in plan.variants}
    assert values == {"ridge", "lasso", "elasticnet"}


def test_two_axis_sweep_cartesian_count() -> None:
    recipe = _base_recipe()
    recipe["path"]["2_preprocessing"]["sweep_axes"] = {
        "scaling_policy": ["none", "standard"],
    }
    recipe["path"]["2_preprocessing"]["fixed_axes"] = {}

    plan = compile_sweep_plan(recipe)

    assert plan.size == 3 * 2
    assert set(plan.axes_swept) == {
        "2_preprocessing.scaling_policy",
        "3_training.model_family",
    }
    seen = {
        (v.axis_values["2_preprocessing.scaling_policy"], v.axis_values["3_training.model_family"])
        for v in plan.variants
    }
    assert len(seen) == 6
    assert plan.governance["schema_version"] == "sweep_governance_v1"
    assert plan.governance["expansion_policy"] == "cartesian_expand_all_then_compile_each_variant"
    assert plan.governance["variant_count"] == 6


def test_layer2_layer3_grid_example_expands_and_compiles_cells() -> None:
    recipe = yaml.safe_load(Path("examples/recipes/layer2-layer3-grid.yaml").read_text())

    plan = compile_sweep_plan(recipe)

    assert plan.size == 12
    assert set(plan.axes_swept) == {
        "2_preprocessing.target_lag_block",
        "2_preprocessing.x_lag_feature_block",
        "3_training.model_family",
    }
    compiled = [compile_recipe_dict(variant.variant_recipe_dict).compiled for variant in plan.variants]
    statuses = {status: sum(item.execution_status == status for item in compiled) for status in {item.execution_status for item in compiled}}
    assert statuses["executable"] == 8
    assert statuses["blocked_by_incompatibility"] == 4
    blocked = [item for item in compiled if item.execution_status == "blocked_by_incompatibility"]
    assert all("model_family='ar'" in " ".join(item.blocked_reasons) for item in blocked)


def test_variant_id_is_stable_across_calls() -> None:
    r1 = _base_recipe()
    r2 = _base_recipe()
    p1 = compile_sweep_plan(r1)
    p2 = compile_sweep_plan(r2)

    assert p1.study_id == p2.study_id
    assert [v.variant_id for v in p1.variants] == [v.variant_id for v in p2.variants]


def test_variant_id_prefix_and_length() -> None:
    plan = compile_sweep_plan(_base_recipe())
    for variant in plan.variants:
        assert variant.variant_id.startswith("v-")
        assert len(variant.variant_id) == len("v-") + 8


def test_same_axis_in_fixed_and_sweep_raises() -> None:
    recipe = _base_recipe()
    recipe["path"]["3_training"]["fixed_axes"]["model_family"] = "ridge"

    with pytest.raises(SweepPlanError, match="appears in both"):
        compile_sweep_plan(recipe)


def test_no_sweep_axes_raises() -> None:
    recipe = _base_recipe()
    recipe["path"]["3_training"].pop("sweep_axes")

    with pytest.raises(SweepPlanError, match="no sweep_axes"):
        compile_sweep_plan(recipe)


def test_empty_sweep_values_raises() -> None:
    recipe = _base_recipe()
    recipe["path"]["3_training"]["sweep_axes"] = {"model_family": []}

    with pytest.raises(SweepPlanError, match="non-empty list"):
        compile_sweep_plan(recipe)


def test_max_variants_exceeded_raises() -> None:
    recipe = _base_recipe()
    recipe["path"]["3_training"]["sweep_axes"] = {
        "model_family": ["ridge", "lasso"],
    }
    recipe["path"]["2_preprocessing"]["sweep_axes"] = {
        "scaling_policy": ["none", "standard"],
    }
    recipe["path"]["2_preprocessing"]["fixed_axes"] = {}

    with pytest.raises(SweepPlanError, match="exceeds max_variants"):
        compile_sweep_plan(recipe, max_variants=3)


def test_max_variants_none_disables_cap() -> None:
    recipe = _base_recipe()
    recipe["path"]["3_training"]["sweep_axes"] = {
        "model_family": ["ridge", "lasso", "elasticnet", "bayesian_ridge"],
    }

    plan = compile_sweep_plan(recipe, max_variants=None)
    assert plan.size == 4


def test_variant_dict_is_single_path() -> None:
    plan = compile_sweep_plan(_base_recipe())
    for variant in plan.variants:
        training = variant.variant_recipe_dict["path"]["3_training"]
        assert "sweep_axes" not in training
        assert training["fixed_axes"]["model_family"] in {"ridge", "lasso", "elasticnet"}
        assert variant.variant_recipe_dict["recipe_id"].endswith(variant.variant_id)


def test_parent_recipe_dict_is_deep_copied() -> None:
    original = _base_recipe()
    snapshot = copy.deepcopy(original)
    plan = compile_sweep_plan(original)

    plan.parent_recipe_dict["path"]["3_training"]["fixed_axes"]["model_family"] = "mutated"

    assert original == snapshot, "compile_sweep_plan must not mutate input dict"


def test_missing_path_raises() -> None:
    with pytest.raises(SweepPlanError, match="missing 'path'"):
        compile_sweep_plan({"recipe_id": "x"})


def test_default_max_variants_is_1000() -> None:
    assert DEFAULT_MAX_VARIANTS == 1000


# --- Nested sweep tests (axis_type.nested_sweep support) ---


def _nested_recipe() -> dict:
    return {
        "recipe_id": "nested-test",
        "path": {
            "0_meta": {"fixed_axes": {"study_scope": "one_target_compare_methods"}},
            "1_data_task": {
                "fixed_axes": {"dataset": "fred_md", "target_structure": "single_target"},
                "leaf_config": {"target": "INDPRO", "horizons": [1]},
            },
            "2_preprocessing": {"fixed_axes": {"scaling_policy": "standard"}},
            "3_training": {
                "fixed_axes": {"framework": "expanding", "benchmark_family": "autoregressive_bic"},
                "nested_sweep_axes": {
                    "model_family": {
                        "ridge": {"hp_space_style": ["paper_fixed_hp", "grid_linear"]},
                        "lasso": {"hp_space_style": ["paper_fixed_hp", "grid_linear", "grid_log"]},
                    },
                },
            },
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
        },
    }


def test_nested_sweep_expands_non_uniform_children() -> None:
    plan = compile_sweep_plan(_nested_recipe())
    # 2 (ridge children) + 3 (lasso children) = 5
    assert plan.size == 5
    assert plan.axes_swept == (
        "3_training.model_family",
        "3_training.hp_space_style",
    )
    for v in plan.variants:
        # every variant pins both parent and child
        assert "3_training.model_family" in v.axis_values
        assert "3_training.hp_space_style" in v.axis_values
        # materialised recipe has nested_sweep_axes cleared
        assert "nested_sweep_axes" not in v.variant_recipe_dict["path"]["3_training"]


def test_nested_sweep_variant_ids_are_stable_and_unique() -> None:
    plan_a = compile_sweep_plan(_nested_recipe())
    plan_b = compile_sweep_plan(_nested_recipe())
    # byte-identical variant_id set across runs
    assert [v.variant_id for v in plan_a.variants] == [v.variant_id for v in plan_b.variants]
    # no collisions
    assert len({v.variant_id for v in plan_a.variants}) == plan_a.size


def test_nested_sweep_combines_with_regular_sweep_cartesian() -> None:
    recipe = _nested_recipe()
    recipe["path"]["2_preprocessing"] = {"sweep_axes": {"scaling_policy": ["standard", "robust"]}}
    plan = compile_sweep_plan(recipe)
    # 2 regular sweep * 5 nested = 10
    assert plan.size == 10
    assert set(plan.axes_swept) == {
        "2_preprocessing.scaling_policy",
        "3_training.model_family",
        "3_training.hp_space_style",
    }


def test_leaf_sweep_axes_materialize_leaf_config_values() -> None:
    recipe = _base_recipe()
    recipe["path"]["3_training"].pop("sweep_axes")
    recipe["path"]["2_preprocessing"]["leaf_sweep_axes"] = {
        "custom_feature_combiner": ["sum_first_two", "spread_factor"],
    }

    plan = compile_sweep_plan(recipe)

    assert plan.size == 2
    assert plan.axes_swept == ("2_preprocessing.leaf_config.custom_feature_combiner",)
    assert plan.governance["leaf_config_axes"] == [
        "2_preprocessing.leaf_config.custom_feature_combiner",
    ]
    assert plan.governance["method_extension_axes"] == [
        "2_preprocessing.leaf_config.custom_feature_combiner",
    ]
    values = {
        variant.axis_values["2_preprocessing.leaf_config.custom_feature_combiner"]
        for variant in plan.variants
    }
    assert values == {"sum_first_two", "spread_factor"}
    for variant in plan.variants:
        preprocessing = variant.variant_recipe_dict["path"]["2_preprocessing"]
        assert "leaf_sweep_axes" not in preprocessing
        assert "leaf_config.custom_feature_combiner" not in preprocessing["fixed_axes"]
        assert preprocessing["leaf_config"]["custom_feature_combiner"] in values


def test_leaf_sweep_axes_reject_duplicate_fixed_leaf_config() -> None:
    recipe = _base_recipe()
    recipe["path"]["3_training"].pop("sweep_axes")
    recipe["path"]["2_preprocessing"]["leaf_config"] = {
        "custom_feature_combiner": "fixed_combo",
    }
    recipe["path"]["2_preprocessing"]["leaf_sweep_axes"] = {
        "custom_feature_combiner": ["swept_combo"],
    }

    with pytest.raises(SweepPlanError, match="appears in both leaf_config and leaf_sweep_axes"):
        compile_sweep_plan(recipe)


def test_nested_sweep_can_bind_leaf_config_only_to_custom_parent() -> None:
    recipe = _base_recipe()
    recipe["path"]["3_training"].pop("sweep_axes")
    recipe["path"]["2_preprocessing"]["nested_sweep_axes"] = {
        "temporal_feature_block": {
            "moving_average_features": {},
            "custom_temporal_features": {
                "leaf_config.custom_temporal_feature_block": [
                    "temporal_spread",
                    "temporal_slope",
                ],
            },
        },
    }

    plan = compile_sweep_plan(recipe)

    assert plan.size == 3
    assert set(plan.axes_swept) == {
        "2_preprocessing.temporal_feature_block",
        "2_preprocessing.leaf_config.custom_temporal_feature_block",
    }
    builtin = [
        variant
        for variant in plan.variants
        if variant.axis_values["2_preprocessing.temporal_feature_block"] == "moving_average_features"
    ]
    assert len(builtin) == 1
    assert "2_preprocessing.leaf_config.custom_temporal_feature_block" not in builtin[0].axis_values
    builtin_leaf = builtin[0].variant_recipe_dict["path"]["2_preprocessing"].get("leaf_config", {})
    assert "custom_temporal_feature_block" not in builtin_leaf

    custom_values = {
        variant.axis_values["2_preprocessing.leaf_config.custom_temporal_feature_block"]
        for variant in plan.variants
        if variant.axis_values["2_preprocessing.temporal_feature_block"] == "custom_temporal_features"
    }
    assert custom_values == {"temporal_spread", "temporal_slope"}
    for variant in plan.variants:
        preprocessing = variant.variant_recipe_dict["path"]["2_preprocessing"]
        assert "nested_sweep_axes" not in preprocessing
        fixed_axes = preprocessing["fixed_axes"]
        assert fixed_axes["temporal_feature_block"] in {
            "moving_average_features",
            "custom_temporal_features",
        }
        if fixed_axes["temporal_feature_block"] == "custom_temporal_features":
            assert preprocessing["leaf_config"]["custom_temporal_feature_block"] in custom_values


def test_layer2_representation_sweep_governance_records_variant_gate_policy() -> None:
    recipe = _base_recipe()
    recipe["path"]["2_preprocessing"]["fixed_axes"] = {}
    recipe["path"]["2_preprocessing"]["sweep_axes"] = {
        "factor_feature_block": ["none", "pca_static_factors", "custom_factors"],
        "feature_selection_semantics": ["select_before_factor", "select_after_factor"],
    }

    plan = compile_sweep_plan(recipe)

    assert plan.size == 18
    governance = plan.governance
    assert governance["invalid_combination_policy"] == "materialize_then_gate_at_variant_compile_or_execute"
    assert governance["co_sweeps_model_and_layer2"] is True
    assert set(governance["layer2_representation_axes"]) == {
        "2_preprocessing.factor_feature_block",
        "2_preprocessing.feature_selection_semantics",
    }
    assert governance["model_axes"] == ["3_training.model_family"]


def test_nested_sweep_parent_cannot_duplicate_fixed_or_sweep() -> None:
    recipe = _nested_recipe()
    recipe["path"]["3_training"]["fixed_axes"]["model_family"] = "ridge"
    with pytest.raises(SweepPlanError, match="nested_sweep parent axis"):
        compile_sweep_plan(recipe)


def test_nested_sweep_child_spec_must_be_single_axis() -> None:
    recipe = _nested_recipe()
    recipe["path"]["3_training"]["nested_sweep_axes"]["model_family"]["ridge"] = {
        "hp_space_style": ["paper_fixed_hp"],
        "max_iter": [100, 500],
    }
    with pytest.raises(SweepPlanError, match="single-key mapping"):
        compile_sweep_plan(recipe)


def test_nested_sweep_rejects_empty_children() -> None:
    recipe = _nested_recipe()
    recipe["path"]["3_training"]["nested_sweep_axes"] = {"model_family": {}}
    with pytest.raises(SweepPlanError, match="non-empty mapping"):
        compile_sweep_plan(recipe)


def test_nested_sweep_alone_without_regular_sweep_is_allowed() -> None:
    recipe = _nested_recipe()
    # no regular sweep_axes anywhere
    assert all("sweep_axes" not in block for block in recipe["path"].values() if isinstance(block, dict))
    plan = compile_sweep_plan(recipe)
    assert plan.size == 5


def test_no_sweep_and_no_nested_raises_sweep_plan_error() -> None:
    recipe = _nested_recipe()
    recipe["path"]["3_training"].pop("nested_sweep_axes")
    with pytest.raises(SweepPlanError, match="no sweep_axes, leaf_sweep_axes, or nested_sweep_axes"):
        compile_sweep_plan(recipe)
