from __future__ import annotations

from copy import deepcopy
from typing import Any

NAMING_LEDGER_VERSION = "registry_naming_v1"

AXIS_NAME_ALIASES: dict[str, str] = {
    "info_set": "information_set_type",
    "dataset_source": "source_adapter",
    "task": "target_structure",
}

AXIS_VALUE_ALIASES: dict[tuple[str, str], str] = {
    ("research_design", "single_path_benchmark"): "single_forecast_run",
    ("research_design", "orchestrated_bundle"): "study_bundle",
    ("research_design", "replication_override"): "replication_recipe",
    ("experiment_unit", "single_target_single_model"): "single_target_single_generator",
    ("experiment_unit", "single_target_model_grid"): "single_target_generator_grid",
    ("information_set_type", "revised"): "final_revised_data",
    ("information_set_type", "pseudo_oos_revised"): "pseudo_oos_on_revised_data",
    ("target_structure", "single_target_point_forecast"): "single_target",
    ("target_structure", "multi_target_point_forecast"): "multi_target",
    ("official_transform_policy", "dataset_tcode"): "apply_official_tcode",
    ("official_transform_policy", "raw_official_frame"): "keep_official_raw_scale",
    ("official_transform_scope", "apply_tcode_to_target"): "target_only",
    ("official_transform_scope", "apply_tcode_to_X"): "predictors_only",
    ("official_transform_scope", "apply_tcode_to_both"): "target_and_predictors",
    ("official_transform_scope", "apply_tcode_to_none"): "none",
    ("contemporaneous_x_rule", "allow_contemporaneous"): "allow_same_period_predictors",
    ("contemporaneous_x_rule", "forbid_contemporaneous"): "forbid_same_period_predictors",
    ("variable_universe", "preselected_core"): "core_variables",
    ("variable_universe", "category_subset"): "category_variables",
    ("variable_universe", "target_specific_subset"): "target_specific_variables",
    ("variable_universe", "handpicked_set"): "explicit_variable_list",
    ("missing_availability", "complete_case_only"): "require_complete_rows",
    ("missing_availability", "available_case"): "keep_available_rows",
    ("missing_availability", "x_impute_only"): "impute_predictors_only",
    ("missing_availability", "zero_fill_before_start"): "zero_fill_leading_predictor_gaps",
    ("raw_missing_policy", "zero_fill_leading_x_before_tcode"): "zero_fill_leading_predictor_missing_before_tcode",
    ("raw_missing_policy", "x_impute_raw"): "impute_raw_predictors",
    ("raw_missing_policy", "drop_rows_with_raw_missing"): "drop_raw_missing_rows",
    ("raw_outlier_policy", "raw_outlier_to_missing"): "set_raw_outliers_to_missing",
    ("target_transform_policy", "tcode_transformed"): "official_tcode_transformed",
    ("x_transform_policy", "dataset_tcode_transformed"): "official_tcode_transformed",
    ("x_transform_policy", "apply_official_tcode_transformed"): "official_tcode_transformed",
    ("tcode_policy", "tcode_only"): "official_tcode_only",
    ("tcode_policy", "tcode_then_extra_preprocess"): "official_tcode_then_extra_preprocess",
    ("tcode_policy", "extra_preprocess_without_tcode"): "extra_preprocess_only",
    ("tcode_policy", "extra_then_tcode"): "extra_preprocess_then_official_tcode",
    ("tcode_policy", "custom_transform_pipeline"): "custom_transform_sequence",
    ("tcode_application_scope", "apply_tcode_to_target"): "target_only",
    ("tcode_application_scope", "apply_tcode_to_X"): "predictors_only",
    ("tcode_application_scope", "apply_tcode_to_both"): "target_and_predictors",
    ("tcode_application_scope", "apply_tcode_to_none"): "none",
    ("preprocess_order", "tcode_only"): "official_tcode_only",
    ("preprocess_order", "tcode_then_extra"): "official_tcode_then_extra",
    ("preprocess_order", "extra_then_tcode"): "extra_preprocess_then_official_tcode",
    ("representation_policy", "tcode_only"): "official_tcode_only",
    ("predictor_family", "handpicked_set"): "explicit_variable_list",
    ("horizon_target_construction", "future_level_y_t_plus_h"): "future_target_level_t_plus_h",
}

RENAMED_AXES: tuple[dict[str, str], ...] = tuple(
    {"legacy_id": legacy, "canonical_id": canonical}
    for legacy, canonical in sorted(AXIS_NAME_ALIASES.items())
)

RENAMED_VALUES: tuple[dict[str, str], ...] = tuple(
    {"axis": axis, "legacy_id": legacy, "canonical_id": canonical}
    for (axis, legacy), canonical in sorted(AXIS_VALUE_ALIASES.items())
)


def canonical_axis_name(axis_name: str) -> str:
    return AXIS_NAME_ALIASES.get(str(axis_name), str(axis_name))


def canonical_axis_value(axis_name: str, value: str) -> str:
    canonical_axis = canonical_axis_name(axis_name)
    return AXIS_VALUE_ALIASES.get((canonical_axis, str(value)), str(value))


def canonicalize_recipe_path(recipe_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a recipe with canonical axis/value IDs in path blocks."""

    payload = deepcopy(recipe_dict)
    path = payload.get("path")
    if not isinstance(path, dict):
        return payload

    for layer_spec in path.values():
        if not isinstance(layer_spec, dict):
            continue
        for block_name in ("fixed_axes", "sweep_axes", "conditional_axes"):
            block = layer_spec.get(block_name)
            if not isinstance(block, dict):
                continue
            canonical_block: dict[str, Any] = {}
            for raw_axis, raw_value in block.items():
                axis = canonical_axis_name(raw_axis)
                if isinstance(raw_value, list):
                    value = [canonical_axis_value(axis, str(item)) for item in raw_value]
                else:
                    value = canonical_axis_value(axis, str(raw_value))
                canonical_block[axis] = value
            layer_spec[block_name] = canonical_block
        derived = layer_spec.get("derived_axes")
        if isinstance(derived, dict):
            layer_spec["derived_axes"] = {
                canonical_axis_name(raw_axis): rule_name for raw_axis, rule_name in derived.items()
            }
    return payload


def rename_ledger() -> dict[str, Any]:
    return {
        "version": NAMING_LEDGER_VERSION,
        "axis_name_aliases": [dict(item) for item in RENAMED_AXES],
        "axis_value_aliases": [dict(item) for item in RENAMED_VALUES],
    }
