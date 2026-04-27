from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from ..execution.importance_dispatch import (
    DEFAULT_IMPORTANCE_SPEC,
    IMPORTANCE_AXIS_NAMES,
    IMPORTANCE_META_AXIS_NAMES,
    LEGACY_IMPORTANCE_METHOD_TO_AXIS,
    LOCAL_IMPORTANCE_METHODS,
    active_importance_methods,
    canonicalize_importance_spec,
)
from ..registry import get_axis_registry

NAVIGATOR_SCHEMA_VERSION = "navigator_view_v1"

_LAYER_LABELS = {
    "0_meta": "Layer 0: study design / execution grammar",
    "1_data_task": "Layer 1: official data frame",
    "2_preprocessing": "Layer 2: representation construction",
    "3_training": "Layer 3: forecast generation",
    "4_evaluation": "Layer 4: evaluation",
    "5_output_provenance": "Layer 5: outputs and provenance",
    "6_stat_tests": "Layer 6: statistical tests",
    "7_importance": "Layer 7: interpretation and importance",
}

_TREE_AXES = {
    "0_meta": (
        "research_design",
        "experiment_unit",
        "failure_policy",
        "reproducibility_mode",
        "compute_mode",
    ),
    "1_data_task": (
        "dataset",
        "source_adapter",
        "frequency",
        "information_set_type",
        "fred_sd_frequency_policy",
        "fred_sd_state_group",
        "fred_sd_variable_group",
        "target_structure",
        "variable_universe",
        "missing_availability",
        "release_lag_rule",
        "contemporaneous_x_rule",
        "raw_missing_policy",
        "raw_outlier_policy",
        "official_transform_policy",
        "official_transform_scope",
    ),
    "2_preprocessing": (
        "fred_sd_mixed_frequency_representation",
        "horizon_target_construction",
        "tcode_policy",
        "x_missing_policy",
        "x_outlier_policy",
        "scaling_policy",
        "target_transform",
        "target_normalization",
        "target_lag_block",
        "x_lag_feature_block",
        "factor_feature_block",
        "level_feature_block",
        "temporal_feature_block",
        "rotation_feature_block",
        "feature_block_combination",
        "feature_selection_policy",
        "feature_selection_semantics",
        "evaluation_scale",
        "feature_builder",
    ),
    "3_training": (
        "model_family",
        "midasr_weight_family",
        "benchmark_family",
        "forecast_type",
        "forecast_object",
        "exogenous_x_path_policy",
        "recursive_x_model_family",
        "framework",
        "outer_window",
        "refit_policy",
        "min_train_size",
        "training_start_rule",
        "search_algorithm",
        "tuning_objective",
        "tuning_budget",
        "validation_location",
        "validation_size_rule",
        "y_lag_count",
    ),
    "4_evaluation": (
        "primary_metric",
        "point_metrics",
        "density_metrics",
        "direction_metrics",
        "relative_metrics",
        "economic_metrics",
        "benchmark_window",
        "benchmark_scope",
        "agg_time",
        "agg_horizon",
        "agg_target",
        "ranking",
        "report_style",
        "regime_definition",
        "regime_use",
        "regime_metrics",
        "decomposition_target",
        "decomposition_order",
        "oos_period",
    ),
    "5_output_provenance": (
        "export_format",
        "saved_objects",
        "provenance_fields",
        "artifact_granularity",
    ),
    "6_stat_tests": (
        "stat_test",
        "equal_predictive",
        "nested",
        "cpa_instability",
        "multiple_model",
        "density_interval",
        "direction",
        "residual_diagnostics",
        "test_scope",
        "dependence_correction",
        "overlap_handling",
    ),
    "7_importance": (
        "importance_method",
        "importance_scope",
        "importance_model_native",
        "importance_model_agnostic",
        "importance_shap",
        "importance_local_surrogate",
        "importance_partial_dependence",
        "importance_grouped",
        "importance_stability",
        "importance_aggregation",
        "importance_output_style",
        "importance_temporal",
        "importance_gradient_path",
    ),
}

_DEFAULT_SELECTIONS = {
    "forecast_type": "direct",
    "forecast_object": "point_mean",
    "fred_sd_frequency_policy": "report_only",
    "fred_sd_state_group": "all_states",
    "fred_sd_variable_group": "all_sd_variables",
    "fred_sd_mixed_frequency_representation": "calendar_aligned_frame",
    "exogenous_x_path_policy": "unavailable",
    "recursive_x_model_family": "none",
    "primary_metric": "msfe",
    "point_metrics": "MSFE",
    "relative_metrics": "relative_MSFE",
    "direction_metrics": "directional_accuracy",
    "density_metrics": "pinball_loss",
    "benchmark_window": "expanding",
    "benchmark_scope": "same_for_all",
    "agg_time": "full_oos_average",
    "agg_horizon": "equal_weight",
    "agg_target": "report_separately_only",
    "ranking": "mean_metric_rank",
    "report_style": "tidy_dataframe",
    "regime_definition": "none",
    "regime_use": "eval_only",
    "regime_metrics": "all_main_metrics_by_regime",
    "decomposition_target": "preprocessing_effect",
    "decomposition_order": "marginal_effect_only",
    "oos_period": "all_oos_data",
    "export_format": "json",
    "saved_objects": "full_bundle",
    "provenance_fields": "full",
    "artifact_granularity": "aggregated",
    "stat_test": "none",
    "equal_predictive": "none",
    "nested": "none",
    "cpa_instability": "none",
    "multiple_model": "none",
    "density_interval": "none",
    "direction": "none",
    "residual_diagnostics": "none",
    "test_scope": "per_target",
    "dependence_correction": "none",
    "overlap_handling": "allow_overlap",
    "importance_method": "none",
    "importance_scope": "global",
    "importance_model_native": "none",
    "importance_model_agnostic": "none",
    "importance_shap": "none",
    "importance_local_surrogate": "none",
    "importance_partial_dependence": "none",
    "importance_grouped": "none",
    "importance_stability": "none",
    "importance_aggregation": "mean_abs",
    "importance_output_style": "ranked_table",
    "importance_temporal": "static_snapshot",
    "importance_gradient_path": "none",
}

_VIRTUAL_AXES = {
    "exogenous_x_path_policy": (
        "unavailable",
        "hold_last_observed",
        "observed_future_x",
        "scheduled_known_future_x",
        "recursive_x_model",
    ),
    "recursive_x_model_family": ("none", "ar1"),
}

_VIRTUAL_AXIS_STATUSES = {
    "exogenous_x_path_policy": {
        "unavailable": "gated_named",
        "hold_last_observed": "operational_narrow",
        "observed_future_x": "operational_narrow",
        "scheduled_known_future_x": "operational_narrow",
        "recursive_x_model": "operational_narrow",
    },
    "recursive_x_model_family": {
        "none": "gated_named",
        "ar1": "operational_narrow",
    },
}

OPERATIONAL_NARROW_CONTRACTS = (
    {
        "axis": "feature_block_set",
        "values": (
            "transformed_x_lags",
            "selected_sparse_x",
            "level_augmented_x",
            "rotation_augmented_x",
            "mixed_blocks",
            "custom_blocks",
        ),
        "owner_layer": "2_preprocessing",
        "contract": "feature_block_set_public_axis_v1",
        "required_companions": (
            "x_lag_feature_block=fixed_x_lags for transformed_x_lags",
            "non-none feature_selection_policy for selected_sparse_x",
            "non-none level_feature_block for level_augmented_x",
            "non-none rotation_feature_block for rotation_augmented_x",
            "at least two active block sources for mixed_blocks",
            "registered custom block or custom combiner for custom_blocks",
        ),
        "pruning_surface": "compiler blocked_reasons and skip_failed_cell manifests",
    },
    {
        "axis": "fred_sd_mixed_frequency_representation",
        "values": (
            "native_frequency_block_payload",
            "mixed_frequency_model_adapter",
        ),
        "owner_layer": "2_preprocessing",
        "contract": "fred_sd_native_frequency_block_payload_v1",
        "required_companions": (
            "dataset includes fred_sd",
            "feature_builder=raw_feature_panel",
            "registered custom model_family or built-in MIDAS model_family",
            "forecast_type=direct",
            "mixed_frequency_model_adapter additionally records fred_sd_mixed_frequency_model_adapter_v1",
        ),
        "pruning_surface": "Navigator compatibility, compiler blocked_reasons, and runtime route guard",
    },
    {
        "axis": "exogenous_x_path_policy",
        "values": (
            "hold_last_observed",
            "observed_future_x",
            "scheduled_known_future_x",
            "recursive_x_model",
        ),
        "owner_layer": "3_training",
        "contract": "exogenous_x_path_contract_v1",
        "required_companions": (
            "forecast_type=iterated",
            "raw-panel feature runtime",
            "target_lag_block=fixed_target_lags",
            "scheduled_known_future_x_columns for scheduled_known_future_x",
            "recursive_x_model_family=ar1 for recursive_x_model",
        ),
        "pruning_surface": "Layer 3 capability matrix and compiler blocked_reasons",
    },
    {
        "axis": "recursive_x_model_family",
        "values": ("ar1",),
        "owner_layer": "3_training",
        "contract": "exogenous_x_path_contract_v1",
        "required_companions": (
            "exogenous_x_path_policy=recursive_x_model",
            "raw-panel iterated point forecast slice",
        ),
        "pruning_surface": "Navigator compatibility and compiler blocked_reasons",
    },
)

_TREE_MODELS = frozenset({"randomforest", "extratrees", "gbm", "xgboost", "lightgbm", "catboost"})
_LINEAR_MODELS = frozenset(
    {
        "ar",
        "ols",
        "ridge",
        "lasso",
        "elasticnet",
        "bayesianridge",
        "huber",
        "adaptivelasso",
        "midas_almon",
        "midasr",
        "midasr_nealmon",
        "quantile_linear",
    }
)
_DEEP_SEQUENCE_MODELS = frozenset({"lstm", "gru", "tcn"})
_FRED_SD_MIXED_FREQUENCY_BUILTIN_MODELS = frozenset({"midas_almon", "midasr", "midasr_nealmon"})
_BUILTIN_MODELS = frozenset(
    {
        "ar",
        "ols",
        "ridge",
        "lasso",
        "elasticnet",
        "bayesianridge",
        "huber",
        "adaptivelasso",
        "svr_linear",
        "svr_rbf",
        "componentwise_boosting",
        "boosting_ridge",
        "boosting_lasso",
        "pcr",
        "pls",
        "factor_augmented_linear",
        "quantile_linear",
        *_FRED_SD_MIXED_FREQUENCY_BUILTIN_MODELS,
        "randomforest",
        "extratrees",
        "gbm",
        "xgboost",
        "lightgbm",
        "catboost",
        "mlp",
        *_DEEP_SEQUENCE_MODELS,
    }
)
_FRED_SD_ADVANCED_MIXED_FREQUENCY = frozenset(
    {"native_frequency_block_payload", "mixed_frequency_model_adapter"}
)
_RAW_PANEL_BUILDERS = frozenset({"raw_feature_panel", "raw_X_only"})
_AUTOREG_BUILDERS = frozenset({"autoreg_lagged_target"})
_QUANTILE_STATS = frozenset({"none", "dm", "dm_hln", "dm_modified"})
_DIRECTION_STATS = frozenset({"none", "pesaran_timmermann", "binomial_hit"})
_DENSITY_INTERVAL_STATS = frozenset(
    {
        "none",
        "PIT_uniformity",
        "berkowitz",
        "kupiec",
        "christoffersen_unconditional",
        "christoffersen_independence",
        "christoffersen_conditional",
        "interval_coverage",
    }
)
_IMPORTANCE_SPLIT_AXES = frozenset(IMPORTANCE_AXIS_NAMES)
_IMPORTANCE_META_AXES = frozenset(IMPORTANCE_META_AXIS_NAMES)
_LOCAL_IMPORTANCE_METHODS = frozenset({"kernel_shap", "lime", "feature_ablation"})
_STAT_TEST_SPLIT_AXES = (
    "equal_predictive",
    "nested",
    "cpa_instability",
    "multiple_model",
    "density_interval",
    "direction",
    "residual_diagnostics",
)
_LEGACY_STAT_TEST_TO_SPLIT = {
    "dm": ("equal_predictive", "dm"),
    "dm_hln": ("equal_predictive", "dm_hln"),
    "dm_modified": ("equal_predictive", "dm_modified"),
    "cw": ("nested", "cw"),
    "enc_new": ("nested", "enc_new"),
    "mse_f": ("nested", "mse_f"),
    "mse_t": ("nested", "mse_t"),
    "cpa": ("cpa_instability", "cpa"),
    "rossi": ("cpa_instability", "rossi"),
    "rolling_dm": ("cpa_instability", "rolling_dm"),
    "reality_check": ("multiple_model", "reality_check"),
    "spa": ("multiple_model", "spa"),
    "mcs": ("multiple_model", "mcs"),
    "pesaran_timmermann": ("direction", "pesaran_timmermann"),
    "binomial_hit": ("direction", "binomial_hit"),
    "mincer_zarnowitz": ("residual_diagnostics", "mincer_zarnowitz"),
    "ljung_box": ("residual_diagnostics", "ljung_box"),
    "arch_lm": ("residual_diagnostics", "arch_lm"),
    "bias_test": ("residual_diagnostics", "bias_test"),
    "diagnostics_full": ("residual_diagnostics", "diagnostics_full"),
}
_HAC_COMPATIBLE_STAT_TESTS = frozenset(
    {"dm_hln", "dm_modified", "cw", "enc_new", "mse_t", "cpa", "spa", "mcs"}
)


def load_recipe(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _recipe_path(recipe: Mapping[str, Any]) -> Mapping[str, Any]:
    return recipe.get("path", {}) if isinstance(recipe.get("path", {}), Mapping) else {}


def _layer_fixed(recipe: Mapping[str, Any], layer: str) -> Mapping[str, Any]:
    layer_payload = _recipe_path(recipe).get(layer, {})
    return layer_payload.get("fixed_axes", {}) if isinstance(layer_payload, Mapping) else {}


def _layer_sweep(recipe: Mapping[str, Any], layer: str) -> Mapping[str, Any]:
    layer_payload = _recipe_path(recipe).get(layer, {})
    return layer_payload.get("sweep_axes", {}) if isinstance(layer_payload, Mapping) else {}


def _layer_leaf(recipe: Mapping[str, Any], layer: str) -> Mapping[str, Any]:
    layer_payload = _recipe_path(recipe).get(layer, {})
    return layer_payload.get("leaf_config", {}) if isinstance(layer_payload, Mapping) else {}


def _selection_map(recipe: Mapping[str, Any]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for layer in _LAYER_LABELS:
        for key, value in _layer_fixed(recipe, layer).items():
            selected[key] = value
        for key, value in _layer_sweep(recipe, layer).items():
            selected[key] = value
        for key, value in _layer_leaf(recipe, layer).items():
            if key not in {"benchmark_config", "training_config"}:
                selected.setdefault(key, value)
    for key, value in _DEFAULT_SELECTIONS.items():
        selected.setdefault(key, value)
    return selected


def canonical_path(recipe: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for layer, label in _LAYER_LABELS.items():
        fixed = dict(_layer_fixed(recipe, layer))
        sweep = dict(_layer_sweep(recipe, layer))
        leaf = dict(_layer_leaf(recipe, layer))
        if fixed or sweep or leaf:
            out[layer] = {
                "label": label,
                "fixed_axes": fixed,
                "sweep_axes": sweep,
                "leaf_config": leaf,
            }
    return out


def _axis_values(axis_name: str, registry: Mapping[str, Any]) -> tuple[str, ...]:
    if axis_name in registry:
        return tuple(registry[axis_name].allowed_values)
    return _VIRTUAL_AXES.get(axis_name, ())


def _axis_status(axis_name: str, value: str, registry: Mapping[str, Any]) -> str:
    if axis_name in registry:
        return str(registry[axis_name].current_status.get(value, "unknown"))
    if axis_name in _VIRTUAL_AXIS_STATUSES:
        return _VIRTUAL_AXIS_STATUSES[axis_name].get(value, "unknown")
    return "unknown"


def _status_disabled_reason(status: str) -> str | None:
    if status in {"operational", "operational_narrow"}:
        return None
    if status == "registry_only":
        return "registered in the grammar, but no current runtime cell is open"
    if status == "future":
        return "future design value; runtime contract is not open"
    if status == "external_plugin":
        return "requires a registered external plugin/callable"
    if status in {"gated_named", "not_supported_yet"}:
        return "named contract exists, but runtime is gated"
    return None


def _selected_stat_tests(
    selected: Mapping[str, Any],
    *,
    override_axis: str | None = None,
    override_value: str | None = None,
) -> dict[str, str]:
    values = {axis: str(selected.get(axis, "none")) for axis in _STAT_TEST_SPLIT_AXES}
    legacy = str(selected.get("stat_test", "none"))
    if legacy != "none" and legacy in _LEGACY_STAT_TEST_TO_SPLIT:
        axis, value = _LEGACY_STAT_TEST_TO_SPLIT[legacy]
        if values.get(axis, "none") == "none":
            values[axis] = value
    if override_axis == "stat_test":
        if legacy != "none" and legacy in _LEGACY_STAT_TEST_TO_SPLIT:
            axis, _ = _LEGACY_STAT_TEST_TO_SPLIT[legacy]
            values[axis] = "none"
        if override_value and override_value != "none" and override_value in _LEGACY_STAT_TEST_TO_SPLIT:
            axis, value = _LEGACY_STAT_TEST_TO_SPLIT[override_value]
            values[axis] = value
    elif override_axis in values and override_value is not None:
        values[str(override_axis)] = override_value
    return {axis: value for axis, value in values.items() if value != "none"}


def _selected_importance_methods(
    selected: Mapping[str, Any],
    *,
    override_axis: str | None = None,
    override_value: str | None = None,
) -> tuple[str, ...]:
    raw = {key: selected.get(key, value) for key, value in DEFAULT_IMPORTANCE_SPEC.items()}
    if override_axis == "importance_method":
        raw = {axis: ("none" if axis in _IMPORTANCE_SPLIT_AXES else value) for axis, value in raw.items()}
    if override_axis in raw and override_value is not None:
        raw[str(override_axis)] = override_value
    return active_importance_methods(canonicalize_importance_spec(raw))


def _selection_has_fred_sd(selected: Mapping[str, Any]) -> bool:
    dataset = str(selected.get("dataset", ""))
    tokens = {token.strip().lower() for token in dataset.replace(",", "+").split("+") if token.strip()}
    return "fred_sd" in tokens


def _compatibility_reason(axis_name: str, value: str, selected: Mapping[str, Any]) -> str | None:
    model = str(selected.get("model_family", ""))
    feature_builder = str(selected.get("feature_builder", ""))
    forecast_type = str(selected.get("forecast_type", "direct"))
    forecast_object = str(selected.get("forecast_object", "point_mean"))
    fred_sd_mixed_frequency = str(
        selected.get("fred_sd_mixed_frequency_representation", "calendar_aligned_frame")
    )
    x_path = str(selected.get("exogenous_x_path_policy", "unavailable"))
    importance = str(selected.get("importance_method", "none"))
    importance_methods = set(_selected_importance_methods(selected))

    if axis_name == "fred_sd_frequency_policy" and value != "report_only" and not _selection_has_fred_sd(selected):
        return "fred_sd_frequency_policy requires dataset to include fred_sd"
    if axis_name == "fred_sd_state_group" and value != "all_states" and not _selection_has_fred_sd(selected):
        return "fred_sd_state_group requires dataset to include fred_sd"
    if axis_name == "fred_sd_variable_group" and value != "all_sd_variables" and not _selection_has_fred_sd(selected):
        return "fred_sd_variable_group requires dataset to include fred_sd"
    if (
        axis_name == "fred_sd_mixed_frequency_representation"
        and value != "calendar_aligned_frame"
        and not _selection_has_fred_sd(selected)
    ):
        return "fred_sd_mixed_frequency_representation requires dataset to include fred_sd"
    if (
        axis_name == "fred_sd_mixed_frequency_representation"
        and model in _FRED_SD_MIXED_FREQUENCY_BUILTIN_MODELS
        and value not in _FRED_SD_ADVANCED_MIXED_FREQUENCY
    ):
        return "selected built-in MIDAS model requires an advanced FRED-SD mixed-frequency representation"
    if axis_name == "fred_sd_mixed_frequency_representation" and value in _FRED_SD_ADVANCED_MIXED_FREQUENCY:
        if feature_builder != "raw_feature_panel":
            return "advanced FRED-SD mixed-frequency representation requires a raw-panel feature builder"
        if model in _BUILTIN_MODELS and model not in _FRED_SD_MIXED_FREQUENCY_BUILTIN_MODELS:
            return "advanced FRED-SD mixed-frequency representation requires a registered custom model or built-in MIDAS model"
        if forecast_type != "direct":
            return "advanced FRED-SD mixed-frequency representation currently supports forecast_type=direct only"

    if axis_name == "feature_builder":
        if fred_sd_mixed_frequency in _FRED_SD_ADVANCED_MIXED_FREQUENCY and value != "raw_feature_panel":
            return "advanced FRED-SD mixed-frequency representation requires a raw-panel feature builder"
        if model in _DEEP_SEQUENCE_MODELS:
            if value == "autoreg_lagged_target":
                return None
            if value == "sequence_tensor":
                return "full multivariate sequence_tensor remains gated; current deep slice is univariate target-history sequence"
            return f"model_family={model} consumes the current univariate sequence/autoreg path, not {value}"
        if model == "ar" and value in _RAW_PANEL_BUILDERS:
            return "model_family=ar is target-lag/autoreg only; raw-panel Z is incompatible"
    if axis_name == "model_family":
        if (
            value in _FRED_SD_MIXED_FREQUENCY_BUILTIN_MODELS
            and fred_sd_mixed_frequency not in _FRED_SD_ADVANCED_MIXED_FREQUENCY
        ):
            return "selected built-in MIDAS model requires an advanced FRED-SD mixed-frequency representation"
        if (
            fred_sd_mixed_frequency in _FRED_SD_ADVANCED_MIXED_FREQUENCY
            and value in _BUILTIN_MODELS
            and value not in _FRED_SD_MIXED_FREQUENCY_BUILTIN_MODELS
        ):
            return "advanced FRED-SD mixed-frequency representation requires a registered custom model or built-in MIDAS model"
        if feature_builder in _RAW_PANEL_BUILDERS and value == "ar":
            return "raw-panel feature builders cannot feed the AR-BIC target-lag generator"
        if feature_builder == "sequence_tensor" and value not in _DEEP_SEQUENCE_MODELS:
            return "sequence_tensor is reserved for sequence/tensor generators"
        if (importance == "tree_shap" or "tree_shap" in importance_methods) and value not in _TREE_MODELS:
            return "importance_method=tree_shap requires a tree model"
        if (importance == "linear_shap" or "linear_shap" in importance_methods) and value not in _LINEAR_MODELS:
            return "importance_method=linear_shap requires a linear estimator"
        if forecast_object == "quantile" and value != "quantile_linear":
            return "forecast_object=quantile currently requires model_family=quantile_linear"
    if axis_name == "midasr_weight_family":
        if model not in {"midasr", "midasr_nealmon"}:
            return "midasr_weight_family applies only to model_family=midasr or midasr_nealmon"
        if model == "midasr_nealmon" and value != "nealmon":
            return "model_family=midasr_nealmon is the compatibility alias for nealmon only"
    if axis_name == "forecast_object":
        if value == "quantile" and model and model != "quantile_linear":
            return "quantile forecasts currently require model_family=quantile_linear"
        if value in {"interval", "density"} and str(selected.get("target_normalization", "none")) != "none":
            return "interval/density payload wrappers currently require target_normalization=none"
    if axis_name == "forecast_type":
        if fred_sd_mixed_frequency in _FRED_SD_ADVANCED_MIXED_FREQUENCY and value != "direct":
            return "advanced FRED-SD mixed-frequency representation currently supports forecast_type=direct only"
    if axis_name == "importance_method":
        methods = _selected_importance_methods(selected, override_axis=axis_name, override_value=value)
        if "tree_shap" in methods and model and model not in _TREE_MODELS:
            return "tree_shap requires a tree model"
        if "linear_shap" in methods and model and model not in _LINEAR_MODELS:
            return "linear_shap requires a linear estimator"
    if axis_name in _IMPORTANCE_SPLIT_AXES:
        methods = _selected_importance_methods(selected, override_axis=axis_name, override_value=value)
        if "tree_shap" in methods and model and model not in _TREE_MODELS:
            return "tree_shap requires a tree model"
        if "linear_shap" in methods and model and model not in _LINEAR_MODELS:
            return "linear_shap requires a linear estimator"
        if "minimal_importance" in methods and feature_builder not in _RAW_PANEL_BUILDERS:
            return "minimal_importance currently requires a raw-panel feature builder"
    if axis_name in _IMPORTANCE_META_AXES:
        methods = _selected_importance_methods(selected)
        default_value = DEFAULT_IMPORTANCE_SPEC[axis_name]
        if not methods and value != default_value:
            return "Layer 7 detail axes are active only when an importance family is selected"
        if axis_name == "importance_scope":
            if value == "global" and methods and set(methods) <= set(_LOCAL_IMPORTANCE_METHODS):
                return "local-only importance methods require importance_scope=local"
            if value == "local" and methods and not (set(methods) & set(_LOCAL_IMPORTANCE_METHODS)):
                return "global-only importance methods require importance_scope=global"
    if axis_name == "exogenous_x_path_policy":
        if forecast_type != "iterated" and value != "unavailable":
            return "future-X path policies apply only when forecast_type=iterated"
        if value != "unavailable" and feature_builder not in _RAW_PANEL_BUILDERS:
            return "raw-panel iterated future-X paths require a raw-panel feature builder"
    if axis_name == "recursive_x_model_family":
        if x_path != "recursive_x_model" and value != "none":
            return "recursive_x_model_family is active only for exogenous_x_path_policy=recursive_x_model"
        if x_path == "recursive_x_model" and value != "ar1":
            return "only recursive_x_model_family=ar1 is currently operational"
    if axis_name == "stat_test":
        if forecast_object == "direction" and value not in _DIRECTION_STATS:
            return "direction forecast objects should use direction-family tests"
        if forecast_object in {"interval", "density"} and value not in {"none"}:
            return "interval/density calibration tests live on the density_interval axis, not legacy stat_test"
        if forecast_object == "quantile" and value not in _QUANTILE_STATS:
            return "quantile tasks should avoid legacy point-forecast-only tests"
        if str(selected.get("overlap_handling", "allow_overlap")) == "evaluate_with_hac":
            active_tests = _selected_stat_tests(selected, override_axis=axis_name, override_value=value)
            incompatible = {test for test in active_tests.values() if test not in _HAC_COMPATIBLE_STAT_TESTS}
            if incompatible:
                return "evaluate_with_hac requires HAC-capable Layer 6 tests"
    if axis_name == "density_interval":
        if forecast_object not in {"interval", "density", "quantile"} and value != "none":
            return "density/interval tests require interval, density, or quantile forecast objects"
    if axis_name == "direction":
        if forecast_object != "direction" and value != "none":
            return "direction tests require forecast_object=direction"
    if axis_name in _STAT_TEST_SPLIT_AXES and str(selected.get("overlap_handling", "allow_overlap")) == "evaluate_with_hac":
        active_tests = _selected_stat_tests(selected, override_axis=axis_name, override_value=value)
        incompatible = {test for test in active_tests.values() if test not in _HAC_COMPATIBLE_STAT_TESTS}
        if incompatible:
            return "evaluate_with_hac requires HAC-capable Layer 6 tests"
    if axis_name == "dependence_correction":
        active_tests = _selected_stat_tests(selected)
        if value in {"nw_hac", "nw_hac_auto", "block_bootstrap"} and not active_tests:
            return "dependence corrections are active only when a Layer 6 test is selected"
        incompatible = {test for test in active_tests.values() if test not in _HAC_COMPATIBLE_STAT_TESTS}
        if value in {"nw_hac", "nw_hac_auto", "block_bootstrap"} and incompatible:
            return "dependence corrections require HAC/bootstrap-compatible Layer 6 tests"
    if axis_name == "overlap_handling":
        active_tests = _selected_stat_tests(selected)
        incompatible = {test for test in active_tests.values() if test not in _HAC_COMPATIBLE_STAT_TESTS}
        if value == "evaluate_with_hac" and incompatible:
            return "evaluate_with_hac requires HAC-capable Layer 6 tests"
    return None


def _canonical_path_effect(layer: str, axis_name: str, value: str) -> str:
    if axis_name in {"exogenous_x_path_policy", "recursive_x_model_family"}:
        return f"path.{layer}.leaf_config.{axis_name} = {value!r}"
    return f"path.{layer}.fixed_axes.{axis_name} = {value!r}"


def navigator_state_engine_spec() -> dict[str, Any]:
    """Serializable rule metadata for the browser-side Navigator state engine."""

    return {
        "schema_version": "navigator_state_engine_v1",
        "default_selections": dict(_DEFAULT_SELECTIONS),
        "status_disabled_reasons": {
            "registry_only": "registered in the grammar, but no current runtime cell is open",
            "future": "future design value; runtime contract is not open",
            "external_plugin": "requires a registered external plugin/callable",
            "gated_named": "named contract exists, but runtime is gated",
            "not_supported_yet": "named contract exists, but runtime is gated",
        },
        "model_groups": {
            "tree_models": sorted(_TREE_MODELS),
            "linear_models": sorted(_LINEAR_MODELS),
            "deep_sequence_models": sorted(_DEEP_SEQUENCE_MODELS),
            "fred_sd_mixed_frequency_builtin_models": sorted(_FRED_SD_MIXED_FREQUENCY_BUILTIN_MODELS),
            "raw_panel_builders": sorted(_RAW_PANEL_BUILDERS),
            "autoreg_builders": sorted(_AUTOREG_BUILDERS),
        },
        "forecast_object_rules": {
            "quantile_model": "quantile_linear",
            "direction_stats": sorted(_DIRECTION_STATS),
            "density_interval_stats": sorted(_DENSITY_INTERVAL_STATS),
            "quantile_stats": sorted(_QUANTILE_STATS),
        },
        "stat_tests": {
            "split_axes": list(_STAT_TEST_SPLIT_AXES),
            "legacy_to_split": {
                key: {"axis": axis, "value": value}
                for key, (axis, value) in sorted(_LEGACY_STAT_TEST_TO_SPLIT.items())
            },
            "hac_compatible": sorted(_HAC_COMPATIBLE_STAT_TESTS),
        },
        "importance": {
            "split_axes": list(IMPORTANCE_AXIS_NAMES),
            "meta_axes": list(IMPORTANCE_META_AXIS_NAMES),
            "default_spec": dict(DEFAULT_IMPORTANCE_SPEC),
            "legacy_to_axis": {
                key: {"axis": axis, "value": value}
                for key, (axis, value) in sorted(LEGACY_IMPORTANCE_METHOD_TO_AXIS.items())
            },
            "local_methods": sorted(LOCAL_IMPORTANCE_METHODS),
        },
    }


def _axis_view(layer: str, axis_name: str, selected: Mapping[str, Any], registry: Mapping[str, Any]) -> dict[str, Any]:
    values = _axis_values(axis_name, registry)
    options: list[dict[str, Any]] = []
    for value in values:
        status = _axis_status(axis_name, value, registry)
        reason = _status_disabled_reason(status)
        compatibility_reason = _compatibility_reason(axis_name, value, selected)
        enabled = reason is None and compatibility_reason is None
        disabled_reason = compatibility_reason or reason
        options.append(
            {
                "value": value,
                "status": status,
                "enabled": enabled,
                "disabled_reason": disabled_reason,
                "canonical_path_effect": _canonical_path_effect(layer, axis_name, value),
            }
        )
    return {
        "layer": layer,
        "layer_label": _LAYER_LABELS[layer],
        "axis": axis_name,
        "selected": selected.get(axis_name),
        "options": options,
    }


def tree_view(recipe: Mapping[str, Any], *, include_layers: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    registry = get_axis_registry()
    selected = _selection_map(recipe)
    layers = include_layers or tuple(_TREE_AXES)
    out: list[dict[str, Any]] = []
    for layer in layers:
        for axis_name in _TREE_AXES.get(layer, ()):
            if axis_name in registry or axis_name in _VIRTUAL_AXES:
                out.append(_axis_view(layer, axis_name, selected, registry))
    return out


def compatibility_view(recipe: Mapping[str, Any]) -> dict[str, Any]:
    selected = _selection_map(recipe)
    active_rules: list[dict[str, str]] = []
    recommendations: list[str] = []
    model = str(selected.get("model_family", ""))
    feature_builder = str(selected.get("feature_builder", ""))
    forecast_object = str(selected.get("forecast_object", "point_mean"))
    importance_methods = set(_selected_importance_methods(selected))
    forecast_type = str(selected.get("forecast_type", "direct"))

    if model in _DEEP_SEQUENCE_MODELS:
        active_rules.append(
            {
                "rule": "deep_model_current_sequence_slice",
                "effect": "keep current univariate target-history sequence/autoreg path; full sequence_tensor remains gated",
            }
        )
    if importance_methods:
        active_rules.append({"rule": "layer7_importance_split_contract", "effect": "split importance-family axes materialize Layer 7 artifacts"})
    if "tree_shap" in importance_methods:
        active_rules.append({"rule": "tree_shap_requires_tree_model", "effect": "model_family restricted to tree generators"})
    if "linear_shap" in importance_methods:
        active_rules.append({"rule": "linear_shap_requires_linear_model", "effect": "model_family restricted to linear estimators"})
    if forecast_object == "quantile":
        active_rules.append({"rule": "quantile_requires_quantile_generator", "effect": "model_family=quantile_linear"})
        recommendations.append("Use quantile-oriented metrics/tests such as pinball/coverage families when those downstream axes are active.")
    if forecast_object == "direction":
        recommendations.append("Use direction-family tests such as pesaran_timmermann or binomial_hit.")
    if forecast_object in {"interval", "density"}:
        recommendations.append("Use density_interval tests; interval/density payloads are baseline wrappers over scalar generators.")
    if str(selected.get("export_format", "json")) in {"parquet", "all"}:
        recommendations.append("Parquet output writes sidecar artifact files in addition to the always-written CSV prediction table.")
    if str(selected.get("regime_definition", "none")) != "none":
        recommendations.append("Regime evaluation is post-forecast evaluation filtering; regime_use beyond eval_only remains a separate runtime gate.")
    if forecast_type == "iterated" and feature_builder in _RAW_PANEL_BUILDERS:
        active_rules.append(
            {
                "rule": "raw_panel_iterated_future_x_path",
                "effect": "leaf_config.exogenous_x_path_policy selects hold, observed, scheduled-known, or recursive-X/ar1 path",
            }
        )
    return {
        "selected": dict(sorted(selected.items())),
        "active_rules": active_rules,
        "recommendations": recommendations,
    }


def _compile_preview(recipe: Mapping[str, Any]) -> dict[str, Any]:
    from ..compiler import CompileValidationError, compile_recipe_dict

    try:
        compiled = compile_recipe_dict(dict(recipe))
    except CompileValidationError as exc:
        return {
            "execution_status": "compile_error",
            "warnings": [str(exc)],
            "blocked_reasons": [str(exc)],
        }
    manifest = compiled.manifest
    return {
        "execution_status": manifest.get("execution_status"),
        "warnings": list(manifest.get("warnings", [])),
        "blocked_reasons": list(manifest.get("blocked_reasons", [])),
        "tree_context": dict(manifest.get("tree_context", {})),
        "layer3_capability_matrix": dict(manifest.get("layer3_capability_matrix", {})),
    }


def build_navigation_view(recipe: Mapping[str, Any], *, include_downstream: bool = True) -> dict[str, Any]:
    layers = tuple(_TREE_AXES) if include_downstream else ("0_meta", "1_data_task", "2_preprocessing", "3_training")
    return {
        "schema_version": NAVIGATOR_SCHEMA_VERSION,
        "recipe_id": recipe.get("recipe_id", ""),
        "canonical_path": canonical_path(recipe),
        "tree": tree_view(recipe, include_layers=layers),
        "compatibility": compatibility_view(recipe),
        "compile_preview": _compile_preview(recipe),
    }


def build_navigation_view_from_yaml(path: str | Path, *, include_downstream: bool = True) -> dict[str, Any]:
    return build_navigation_view(load_recipe(path), include_downstream=include_downstream)


def resolve_yaml_path(path: str | Path) -> dict[str, Any]:
    from ..compiler import compile_recipe_yaml

    compiled = compile_recipe_yaml(path)
    return {
        "input_yaml_path": str(path),
        "execution_status": compiled.compiled.execution_status,
        "manifest": compiled.manifest,
        "navigation": build_navigation_view_from_yaml(path),
    }
