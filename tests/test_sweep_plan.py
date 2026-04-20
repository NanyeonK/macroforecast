"""Tests for macrocast.compiler.sweep_plan (Phase 1 sub-task 01.1 / 01.7)."""

from __future__ import annotations

import copy

import pytest

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
            "0_meta": {"fixed_axes": {"research_design": "controlled_variation"}},
            "1_data_task": {
                "fixed_axes": {"dataset": "fred_md", "task": "single_target_point_forecast"},
                "leaf_config": {"target": "INDPRO", "horizons": [1]},
            },
            "2_preprocessing": {"fixed_axes": {"scaling_policy": "standard"}},
            "3_training": {
                "fixed_axes": {"framework": "expanding", "benchmark_family": "ar_bic"},
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
        "model_family": ["ridge", "lasso", "elasticnet", "bayesianridge"],
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
            "0_meta": {"fixed_axes": {"research_design": "controlled_variation"}},
            "1_data_task": {
                "fixed_axes": {"dataset": "fred_md", "task": "single_target_point_forecast"},
                "leaf_config": {"target": "INDPRO", "horizons": [1]},
            },
            "2_preprocessing": {"fixed_axes": {"scaling_policy": "standard"}},
            "3_training": {
                "fixed_axes": {"framework": "expanding", "benchmark_family": "ar_bic"},
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
    with pytest.raises(SweepPlanError, match="no sweep_axes or nested_sweep_axes"):
        compile_sweep_plan(recipe)
