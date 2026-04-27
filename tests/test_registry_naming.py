from __future__ import annotations

from macrocast.compiler import compile_recipe_dict
from macrocast.defaults import build_default_recipe_dict
from macrocast.registry.naming import (
    canonical_axis_value,
    canonicalize_recipe_path,
    rename_ledger,
)


def test_stage0_legacy_value_aliases_canonicalize() -> None:
    assert canonical_axis_value("research_design", "single_path_benchmark") == "single_forecast_run"
    assert canonical_axis_value("research_design", "orchestrated_bundle") == "study_bundle"
    assert canonical_axis_value("research_design", "replication_override") == "replication_recipe"
    assert (
        canonical_axis_value("experiment_unit", "single_target_single_model")
        == "single_target_single_generator"
    )
    assert (
        canonical_axis_value("experiment_unit", "single_target_model_grid")
        == "single_target_generator_grid"
    )


def test_canonicalize_recipe_path_rewrites_legacy_stage0_values() -> None:
    recipe = build_default_recipe_dict(dataset="fred_md", target="INDPRO", start="1960-01-01", end="1970-01-01")
    recipe["path"]["0_meta"]["fixed_axes"]["research_design"] = "single_path_benchmark"
    recipe["path"]["0_meta"]["fixed_axes"]["experiment_unit"] = "single_target_model_grid"

    canonical = canonicalize_recipe_path(recipe)

    assert canonical["path"]["0_meta"]["fixed_axes"]["research_design"] == "single_forecast_run"
    assert canonical["path"]["0_meta"]["fixed_axes"]["experiment_unit"] == "single_target_generator_grid"


def test_legacy_stage0_recipe_compiles_to_canonical_ids() -> None:
    recipe = build_default_recipe_dict(dataset="fred_md", target="INDPRO", start="1960-01-01", end="1970-01-01")
    recipe["path"]["0_meta"]["fixed_axes"]["research_design"] = "single_path_benchmark"
    recipe["path"]["0_meta"]["fixed_axes"]["experiment_unit"] = "single_target_single_model"

    result = compile_recipe_dict(recipe)

    assert result.compiled.stage0.research_design == "single_forecast_run"
    assert result.compiled.stage0.experiment_unit == "single_target_single_generator"
    assert result.manifest["tree_context"]["fixed_axes"]["research_design"] == "single_forecast_run"
    assert result.manifest["tree_context"]["fixed_axes"]["experiment_unit"] == "single_target_single_generator"


def test_rename_ledger_lists_stage0_aliases() -> None:
    aliases = {
        (item["axis"], item["legacy_id"]): item["canonical_id"]
        for item in rename_ledger()["axis_value_aliases"]
    }

    assert aliases[("research_design", "single_path_benchmark")] == "single_forecast_run"
    assert aliases[("experiment_unit", "single_target_model_grid")] == "single_target_generator_grid"

