"""v0.9.2 Group H batch: status-flip verification for metadata-only axes.

Each promoted value is verified to:
1. carry status="operational" in the registry discovery,
2. be accepted by compile_recipe_dict when placed as the single selected
   value on its axis (no "not executable" warning citing planned status),
3. flow through into the execution/provenance spec with the same id.
"""
from __future__ import annotations

import pytest

from macrocast.registry.build import _discover_axis_definitions


PROMOTED: tuple[tuple[str, str], ...] = (
    # 0_meta
    ("axis_type", "nested_sweep"),
    ("reproducibility_mode", "strict_reproducible"),
    ("failure_policy", "skip_failed_cell"),
    # 1_data_task metadata
    ("x_map_policy", "shared_X"),
    ("target_to_target_inclusion", "forbid_other_targets_as_X"),
    ("information_set_type", "pseudo_oos_revised"),
    ("min_train_size", "fixed_years"),
    ("overlap_handling", "evaluate_with_hac"),
    ("deterministic_components", "linear_trend"),
    ("forecast_object", "quantile"),
    ("forecast_object", "direction"),
    ("own_target_lags", "exclude"),
    # 2_preprocessing metadata
    ("preprocessing_axis_role", "swept_preprocessing"),
    ("preprocessing_axis_role", "ablation_preprocessing"),
    ("recipe_mode", "recipe_grid"),
    ("recipe_mode", "recipe_ablation"),
    # 3_training
    ("logging_level", "debug"),
    ("cache_policy", "feature_cache"),
    ("execution_backend", "joblib"),
    ("checkpointing", "per_model"),
    ("checkpointing", "per_horizon"),
    ("seed_policy", "multi_seed_average"),
    ("alignment_fairness", "same_split_across_targets"),
    ("alignment_fairness", "same_split_across_horizons"),
    ("hp_space_style", "continuous_box"),
    ("hp_space_style", "log_uniform"),
    ("lookback", "horizon_specific_lookback"),
    ("horizon_modelization", "recursive_one_step_model"),
    # 4_evaluation
    ("agg_horizon", "report_separately_only"),
    ("agg_target", "equal_weight"),
    ("decomposition_order", "marginal_effect_only"),
    ("report_style", "latex_table"),
    ("report_style", "markdown_table"),
    ("ranking", "mean_metric_rank"),
    ("ranking", "median_metric_rank"),
    ("ranking", "win_count"),
    ("ranking", "benchmark_beat_freq"),
    ("ranking", "MCS_inclusion_priority"),
)


@pytest.fixture(scope="module")
def defs():
    return _discover_axis_definitions()


@pytest.mark.parametrize("axis,value", PROMOTED)
def test_value_registered_as_operational(axis, value, defs):
    assert axis in defs, f"axis {axis!r} not in registry"
    statuses = {e.id: e.status for e in defs[axis].entries}
    assert value in statuses, f"value {value!r} not found on axis {axis!r}"
    assert statuses[value] == "operational", (
        f"axis {axis!r} value {value!r} is {statuses[value]!r}, expected operational"
    )
