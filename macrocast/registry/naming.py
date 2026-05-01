from __future__ import annotations

from copy import deepcopy
from typing import Any

AXIS_NAME_ALIASES: dict[str, str] = {}

AXIS_VALUE_ALIASES: dict[tuple[str, str], str] = {
    ("primary_metric", "msfe"): "mse",
    ("primary_metric", "relative_msfe"): "relative_mse",
    ("primary_metric", "oos_r2"): "r2_oos",
    ("primary_metric", "csfe"): "mse",
    ("point_metrics", "msfe"): "mse",
    ("relative_metrics", "relative_msfe"): "relative_mse",
    ("relative_metrics", "oos_r2"): "r2_oos",
    ("oos_period", "all_oos_data"): "full_oos",
    ("oos_period", "recession_only_oos"): "full_oos",
    ("oos_period", "expansion_only_oos"): "full_oos",
    ("benchmark_window", "expanding"): "full_oos",
    ("benchmark_window", "fixed"): "full_oos",
    ("benchmark_scope", "same_for_all"): "all_targets_horizons",
    ("benchmark_scope", "target_specific"): "per_target_horizon",
    ("benchmark_scope", "horizon_specific"): "per_target_horizon",
    ("agg_time", "full_out_of_sample_average"): "mean",
    ("agg_horizon", "equal_weight"): "pool_horizons",
    ("agg_target", "report_separately_only"): "per_target_separate",
    ("ranking", "mean_metric_rank"): "by_average_rank",
    ("ranking", "median_metric_rank"): "by_average_rank",
    ("ranking", "win_count"): "by_primary_metric",
    ("ranking", "benchmark_beat_frequency"): "by_relative_metric",
    ("ranking", "mcs_inclusion_priority"): "mcs_inclusion",
    ("report_style", "tidy_dataframe"): "single_table",
    ("regime_use", "evaluation_only"): "pooled",
    ("decomposition_target", "preprocessing_effect"): "none",
    ("decomposition_order", "marginal_effect_only"): "marginal",
    ("saved_objects", "full_bundle"): "forecasts",
    ("saved_objects", "predictions_and_metrics"): "forecasts",
    ("saved_objects", "predictions_only"): "forecasts",
    ("provenance_fields", "full"): "recipe_yaml_full",
    ("provenance_fields", "standard"): "recipe_yaml_full",
    ("artifact_granularity", "aggregated"): "per_cell",
    ("dependence_correction", "nw_hac"): "newey_west",
    ("dependence_correction", "nw_hac_auto"): "andrews",
    ("dependence_correction", "block_bootstrap"): "newey_west",
    ("overlap_handling", "evaluate_with_hac"): "nw_with_h_minus_1_lag",
    ("equal_predictive_test", "dm"): "dm_diebold_mariano",
    ("equal_predictive_test", "dm_hln"): "dm_diebold_mariano",
    ("equal_predictive_test", "dm_modified"): "dm_diebold_mariano",
    ("compute_mode", "parallel_by_model"): "parallel",
    ("compute_mode", "parallel_by_horizon"): "parallel",
    ("compute_mode", "parallel_by_target"): "parallel",
    ("compute_mode", "parallel_by_oos_date"): "parallel",
    ("failure_policy", "skip_failed_cell"): "continue_on_failure",
    ("failure_policy", "skip_failed_model"): "continue_on_failure",
    ("failure_policy", "save_partial_results"): "continue_on_failure",
    ("failure_policy", "warn_only"): "continue_on_failure",
    ("reproducibility_mode", "strict_reproducible"): "seeded_reproducible",
    ("reproducibility_mode", "best_effort"): "seeded_reproducible",
}

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
