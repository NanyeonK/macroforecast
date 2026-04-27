"""1.2 Task & Target cleanup — dropped-value and demoted-value compile tests.

Each dropped registry value must now raise `CompileValidationError` when it
appears on a recipe. Each value demoted to `registry_only` must still compile
but produce `execution_status=not_supported` so end-to-end execution is gated
until a concrete runner lands.
"""
from __future__ import annotations

import pytest

from macrocast.compiler.build import compile_recipe_dict
from macrocast.compiler.errors import CompileValidationError


def _base_recipe(overrides_1_data_task: dict[str, str] | None = None) -> dict:
    axes_1 = {
        "dataset": "fred_md",
        "info_set": "revised",
        "task": "single_target_point_forecast",
    }
    if overrides_1_data_task:
        axes_1.update(overrides_1_data_task)
    return {
        "recipe_id": "stage1-12-cleanup-test",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_forecast_run"}},
            "1_data_task": {
                "fixed_axes": axes_1,
                "leaf_config": {"target": "INDPRO", "horizons": [1]},
            },
            "2_preprocessing": {"fixed_axes": {
                "target_transform_policy": "raw_level",
                "x_transform_policy": "raw_level",
                "tcode_policy": "raw_only",
                "target_missing_policy": "none",
                "x_missing_policy": "none",
                "target_outlier_policy": "none",
                "x_outlier_policy": "none",
                "scaling_policy": "none",
                "dimensionality_reduction_policy": "none",
                "feature_selection_policy": "none",
                "preprocess_order": "none",
                "preprocess_fit_scope": "not_applicable",
                "inverse_transform_policy": "none",
                "evaluation_scale": "raw_level",
            }},
            "3_training": {"fixed_axes": {
                "framework": "expanding",
                "benchmark_family": "zero_change",
                "feature_builder": "autoreg_lagged_target",
                "model_family": "ar",
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {
                "manifest_mode": "full",
                "benchmark_config": {"minimum_train_size": 5},
            }},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }


# Values removed from the registry entirely — compile must reject.
DROPPED: tuple[tuple[str, str], ...] = (
    ("forecast_type", "dirrec"),
    ("forecast_type", "mimo"),
    ("forecast_type", "seq2seq"),
    ("forecast_object", "turning_point"),
    ("forecast_object", "regime_probability"),
    ("forecast_object", "event_probability"),
    ("horizon_target_construction", "annualized_growth_to_h"),
    ("horizon_target_construction", "realized_future_average"),
    ("horizon_target_construction", "future_sum"),
    ("horizon_target_construction", "future_indicator"),
)


@pytest.mark.parametrize("axis,value", DROPPED)
def test_dropped_value_is_rejected(axis: str, value: str) -> None:
    with pytest.raises(CompileValidationError):
        compile_recipe_dict(_base_recipe({axis: value}))


# Values promoted to operational Layer 3 stepwise execution.
PROMOTED: tuple[tuple[str, str], ...] = (
    ("horizon_target_construction", "path_average_growth_1_to_h"),
    ("horizon_target_construction", "path_average_difference_1_to_h"),
    ("horizon_target_construction", "path_average_log_growth_1_to_h"),
    ("forecast_object", "direction"),
    ("forecast_object", "interval"),
    ("forecast_object", "density"),
)


@pytest.mark.parametrize("axis,value", PROMOTED)
def test_promoted_value_is_executable(axis: str, value: str) -> None:
    recipe = _base_recipe({axis: value})
    result = compile_recipe_dict(recipe)
    assert result.compiled.execution_status == "executable"


@pytest.mark.parametrize(
    "value",
    (
        "average_growth_1_to_h",
        "average_difference_1_to_h",
        "average_log_growth_1_to_h",
    ),
)
def test_direct_average_target_construction_is_supported(value: str) -> None:
    result = compile_recipe_dict(_base_recipe({"horizon_target_construction": value}))
    assert result.compiled.execution_status == "executable"


def test_target_to_target_inclusion_axis_dropped() -> None:
    with pytest.raises(CompileValidationError, match="target_to_target_inclusion"):
        compile_recipe_dict(_base_recipe({"target_to_target_inclusion": "forbid_other_targets_as_X"}))
