from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

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
        "ranking",
    ),
    "6_stat_tests": (
        "stat_test",
        "equal_predictive",
        "nested",
        "multiple_model",
        "direction",
        "density_interval",
        "residual_diagnostics",
        "cpa_instability",
        "dependence_correction",
    ),
    "7_importance": (
        "importance_method",
        "importance_shap",
        "model_native",
        "model_agnostic",
        "partial_dependence",
        "grouped",
        "stability",
    ),
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
        "quantile_linear",
    }
)
_DEEP_SEQUENCE_MODELS = frozenset({"lstm", "gru", "tcn"})
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
    selected.setdefault("forecast_type", _layer_fixed(recipe, "3_training").get("forecast_type", "direct"))
    selected.setdefault("forecast_object", _layer_fixed(recipe, "3_training").get("forecast_object", "point_mean"))
    selected.setdefault("exogenous_x_path_policy", _layer_leaf(recipe, "3_training").get("exogenous_x_path_policy", "unavailable"))
    selected.setdefault("recursive_x_model_family", _layer_leaf(recipe, "3_training").get("recursive_x_model_family", "none"))
    selected.setdefault("importance_method", _layer_fixed(recipe, "7_importance").get("importance_method", "none"))
    selected.setdefault("stat_test", _layer_fixed(recipe, "6_stat_tests").get("stat_test", "none"))
    selected.setdefault("primary_metric", _layer_fixed(recipe, "4_evaluation").get("primary_metric", "msfe"))
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
    if axis_name == "exogenous_x_path_policy":
        return "operational_narrow" if value != "unavailable" else "gated_named"
    if axis_name == "recursive_x_model_family":
        return "operational_narrow" if value == "ar1" else "gated_named"
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


def _compatibility_reason(axis_name: str, value: str, selected: Mapping[str, Any]) -> str | None:
    model = str(selected.get("model_family", ""))
    feature_builder = str(selected.get("feature_builder", ""))
    forecast_type = str(selected.get("forecast_type", "direct"))
    forecast_object = str(selected.get("forecast_object", "point_mean"))
    x_path = str(selected.get("exogenous_x_path_policy", "unavailable"))
    importance = str(selected.get("importance_method", "none"))

    if axis_name == "feature_builder":
        if model in _DEEP_SEQUENCE_MODELS:
            if value == "autoreg_lagged_target":
                return None
            if value == "sequence_tensor":
                return "full multivariate sequence_tensor remains gated; current deep slice is univariate target-history sequence"
            return f"model_family={model} consumes the current univariate sequence/autoreg path, not {value}"
        if model == "ar" and value in _RAW_PANEL_BUILDERS:
            return "model_family=ar is target-lag/autoreg only; raw-panel Z is incompatible"
    if axis_name == "model_family":
        if feature_builder in _RAW_PANEL_BUILDERS and value == "ar":
            return "raw-panel feature builders cannot feed the AR-BIC target-lag generator"
        if feature_builder == "sequence_tensor" and value not in _DEEP_SEQUENCE_MODELS:
            return "sequence_tensor is reserved for sequence/tensor generators"
        if importance == "tree_shap" and value not in _TREE_MODELS:
            return "importance_method=tree_shap requires a tree model"
        if importance == "linear_shap" and value not in _LINEAR_MODELS:
            return "importance_method=linear_shap requires a linear estimator"
        if forecast_object == "quantile" and value != "quantile_linear":
            return "forecast_object=quantile currently requires model_family=quantile_linear"
    if axis_name == "forecast_object":
        if value == "quantile" and model and model != "quantile_linear":
            return "quantile forecasts currently require model_family=quantile_linear"
        if value in {"interval", "density"} and str(selected.get("target_normalization", "none")) != "none":
            return "interval/density payload wrappers currently require target_normalization=none"
    if axis_name == "importance_method":
        if value == "tree_shap" and model and model not in _TREE_MODELS:
            return "tree_shap requires a tree model"
        if value == "linear_shap" and model and model not in _LINEAR_MODELS:
            return "linear_shap requires a linear estimator"
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
    if axis_name == "density_interval":
        if forecast_object not in {"interval", "density", "quantile"} and value != "none":
            return "density/interval tests require interval, density, or quantile forecast objects"
    return None


def _canonical_path_effect(layer: str, axis_name: str, value: str) -> str:
    if axis_name in {"exogenous_x_path_policy", "recursive_x_model_family"}:
        return f"path.{layer}.leaf_config.{axis_name} = {value!r}"
    return f"path.{layer}.fixed_axes.{axis_name} = {value!r}"


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
    importance = str(selected.get("importance_method", "none"))
    forecast_type = str(selected.get("forecast_type", "direct"))

    if model in _DEEP_SEQUENCE_MODELS:
        active_rules.append(
            {
                "rule": "deep_model_current_sequence_slice",
                "effect": "keep current univariate target-history sequence/autoreg path; full sequence_tensor remains gated",
            }
        )
    if importance == "tree_shap":
        active_rules.append({"rule": "tree_shap_requires_tree_model", "effect": "model_family restricted to tree generators"})
    if importance == "linear_shap":
        active_rules.append({"rule": "linear_shap_requires_linear_model", "effect": "model_family restricted to linear estimators"})
    if forecast_object == "quantile":
        active_rules.append({"rule": "quantile_requires_quantile_generator", "effect": "model_family=quantile_linear"})
        recommendations.append("Use quantile-oriented metrics/tests such as pinball/coverage families when those downstream axes are active.")
    if forecast_object == "direction":
        recommendations.append("Use direction-family tests such as pesaran_timmermann or binomial_hit.")
    if forecast_object in {"interval", "density"}:
        recommendations.append("Use density_interval tests; interval/density payloads are baseline wrappers over scalar generators.")
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
