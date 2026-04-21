from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .errors import CompileValidationError
from .types import CompiledRecipeSpec, CompileResult
from ..execution import execute_recipe
from ..preprocessing import (
    build_preprocess_contract,
    check_preprocess_governance,
    is_operational_preprocess_contract,
    preprocess_to_dict,
)
from ..recipes import build_recipe_spec, build_run_spec
from ..registry import AxisSelection, get_axis_registry, get_canonical_layer_order
from ..registry.stage0.experiment_unit import derive_experiment_unit_default, get_experiment_unit_entry
from ..design import build_design_frame, resolve_route_owner, design_to_dict

_ALLOWED_SELECTION_MODES = ("fixed_axes", "sweep_axes", "conditional_axes", "leaf_config")

_AXIS_NAME_ALIASES = {
    "info_set": "information_set_type",
}

_AXIS_VALUE_ALIASES = {
    ("information_set_type", "real_time"): "real_time_vintage",
    ("evaluation_scale", "raw_level"): "original_scale",
}

_DATASET_DEFAULT_FREQUENCY = {
    "fred_md": "monthly",
    "fred_qd": "quarterly",
    "fred_sd": "monthly",
}


def load_recipe_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _normalize_layer_spec(layer_spec: dict[str, Any] | None) -> dict[str, Any]:
    layer_spec = layer_spec or {}
    return {key: dict(layer_spec.get(key, {})) for key in _ALLOWED_SELECTION_MODES}


def _canonical_axis_name(axis_name: str) -> str:
    return _AXIS_NAME_ALIASES.get(axis_name, axis_name)


def _canonical_axis_value(axis_name: str, value: str) -> str:
    return _AXIS_VALUE_ALIASES.get((axis_name, value), value)


def _build_axis_selections(recipe_dict: dict[str, Any]) -> tuple[AxisSelection, ...]:
    registry = get_axis_registry()
    path = recipe_dict.get("path", {})
    selections: list[AxisSelection] = []
    for layer in get_canonical_layer_order():
        layer_spec = _normalize_layer_spec(path.get(layer))
        for selection_mode in ("fixed_axes", "sweep_axes", "conditional_axes"):
            mode_name = selection_mode.replace("_axes", "")
            for raw_axis_name, raw_value in layer_spec[selection_mode].items():
                axis_name = _canonical_axis_name(raw_axis_name)
                if axis_name not in registry:
                    raise CompileValidationError(f"unknown registry axis {raw_axis_name!r}")
                entry = registry[axis_name]
                raw_values = tuple(raw_value) if isinstance(raw_value, list) else (raw_value,)
                values = tuple(_canonical_axis_value(axis_name, str(value)) for value in raw_values)
                for value in values:
                    if value not in entry.allowed_values:
                        raise CompileValidationError(
                            f"axis {raw_axis_name!r} received unknown value {value!r}"
                        )
                selections.append(
                    AxisSelection(
                        axis_name=axis_name,
                        layer=layer,
                        selection_mode=mode_name,
                        selected_values=values,
                        selected_status={value: entry.current_status[value] for value in values},
                    )
                )
    return tuple(selections)


def _rule_experiment_unit_default(
    *,
    selection_map: dict[str, AxisSelection],
    leaf_config: dict[str, Any],
) -> str:
    """Derivation rule: default experiment_unit from recipe shape.

    Thin wrapper around macrocast.registry.stage0.experiment_unit.
    derive_experiment_unit_default. Registered under the rule key
    experiment_unit_default in :data:`DERIVATION_RULES`.
    """

    def _sv(name: str, default: str | None = None) -> str | None:
        if name in selection_map:
            return selection_map[name].selected_values[0]
        return default

    research_design = _sv("research_design", "single_path_benchmark")
    task = leaf_config.get("task", "single_target_point_forecast")
    model_sel = selection_map.get("model_family")
    feature_sel = selection_map.get("feature_builder")
    return derive_experiment_unit_default(
        research_design=research_design or "single_path_benchmark",
        task=task,
        model_axis_mode=model_sel.selection_mode if model_sel else "fixed",
        feature_axis_mode=feature_sel.selection_mode if feature_sel else "fixed",
        wrapper_family=leaf_config.get("wrapper_family"),
    )


DERIVATION_RULES: dict[str, Any] = {
    "experiment_unit_default": _rule_experiment_unit_default,
}


def _resolve_derived_axes(
    recipe_dict: dict[str, Any],
    selection_map: dict[str, AxisSelection],
    leaf_config: dict[str, Any],
) -> list[AxisSelection]:
    """Parse recipe's derived_axes sections and resolve each via a registered rule.

    Shape: derived_axes: {axis_name: rule_name} at any layer block.
    Conflict if the axis also appears in fixed/sweep/conditional; unknown
    rule raises CompileValidationError.
    """
    registry = get_axis_registry()
    additions: list[AxisSelection] = []
    for layer in get_canonical_layer_order():
        layer_block = recipe_dict.get("path", {}).get(layer) or {}
        if not isinstance(layer_block, dict):
            continue
        derived = layer_block.get("derived_axes") or {}
        if not isinstance(derived, dict):
            raise CompileValidationError(
                f"layer {layer!r}: derived_axes must be a mapping of axis_name -> rule_name"
            )
        for raw_axis_name, rule_name in derived.items():
            axis_name = _canonical_axis_name(raw_axis_name)
            if axis_name not in registry:
                raise CompileValidationError(
                    f"layer {layer!r}: unknown axis {raw_axis_name!r} in derived_axes"
                )
            if axis_name in selection_map:
                raise CompileValidationError(
                    f"axis {raw_axis_name!r} declared as derived but also appears "
                    "in fixed/sweep/conditional_axes"
                )
            if not isinstance(rule_name, str) or rule_name not in DERIVATION_RULES:
                raise CompileValidationError(
                    f"unknown derivation rule {rule_name!r} for axis {raw_axis_name!r}"
                )
            rule = DERIVATION_RULES[rule_name]
            value = rule(selection_map=selection_map, leaf_config=leaf_config)
            entry = registry[axis_name]
            if value not in entry.allowed_values:
                raise CompileValidationError(
                    f"derivation rule {rule_name!r} produced value {value!r} which is "
                    f"not an allowed value of axis {raw_axis_name!r}"
                )
            additions.append(
                AxisSelection(
                    axis_name=axis_name,
                    layer=layer,
                    selection_mode="derived",
                    selected_values=(value,),
                    selected_status={value: entry.current_status[value]},
                )
            )
    return additions


def _leaf_config(recipe_dict: dict[str, Any]) -> dict[str, Any]:
    path = recipe_dict.get("path", {})
    leaf: dict[str, Any] = {}
    for layer in get_canonical_layer_order():
        leaf.update(_normalize_layer_spec(path.get(layer))["leaf_config"])
    return leaf


def _ensure_unique_axis_selections(selections: tuple[AxisSelection, ...]) -> None:
    seen: dict[str, AxisSelection] = {}
    for selection in selections:
        if selection.axis_name in seen:
            previous = seen[selection.axis_name]
            if previous.selected_values != selection.selected_values or previous.selection_mode != selection.selection_mode or previous.layer != selection.layer:
                raise CompileValidationError(
                    f"axis {selection.axis_name!r} was specified more than once through canonical/legacy aliases"
                )
        else:
            seen[selection.axis_name] = selection


def _selection_map(selections: tuple[AxisSelection, ...]) -> dict[str, AxisSelection]:
    return {selection.axis_name: selection for selection in selections}


def _selection_value(selection_map: dict[str, AxisSelection], axis_name: str, default: str | None = None) -> str:
    if axis_name not in selection_map:
        if default is None:
            raise CompileValidationError(f"missing required axis {axis_name!r}")
        return default
    values = selection_map[axis_name].selected_values
    if len(values) != 1:
        raise CompileValidationError(f"axis {axis_name!r} must be fixed for direct single-run compilation")
    return values[0]


def _build_preprocess_contract(selection_map: dict[str, AxisSelection]) -> Any:
    required = {
        "target_transform_policy",
        "x_transform_policy",
        "tcode_policy",
        "target_missing_policy",
        "x_missing_policy",
        "target_outlier_policy",
        "x_outlier_policy",
        "scaling_policy",
        "dimensionality_reduction_policy",
        "feature_selection_policy",
        "preprocess_order",
        "preprocess_fit_scope",
        "inverse_transform_policy",
        "evaluation_scale",
    }
    defaults = {
        "representation_policy": "raw_only",
        "preprocessing_axis_role": "fixed_preprocessing",
        "tcode_application_scope": "apply_tcode_to_none",
        "target_transform": "level",
        "target_normalization": "none",
        "target_domain": "unconstrained",
        "scaling_scope": "columnwise",
        "additional_preprocessing": "none",
        "x_lag_creation": "no_x_lags",
        "feature_grouping": "none",
        "recipe_mode": "fixed_recipe",
    }
    missing = sorted(axis for axis in required if axis not in selection_map)
    if missing:
        raise CompileValidationError(f"preprocessing layer missing required axes: {missing}")
    payload = {axis: selection_map[axis].selected_values[0] for axis in required}
    payload.update({axis: _selection_value(selection_map, axis, default=value) for axis, value in defaults.items()})
    try:
        contract = build_preprocess_contract(**payload)
        check_preprocess_governance(
            contract,
            preprocessing_sweep=any(
                selection.layer == "2_preprocessing" and selection.selection_mode == "sweep"
                for selection in selection_map.values()
            ),
            model_sweep=(
                "model_family" in selection_map and selection_map["model_family"].selection_mode == "sweep"
            ),
        )
    except Exception as exc:
        raise CompileValidationError(str(exc)) from exc
    return contract


def _benchmark_spec(selection_map: dict[str, AxisSelection], leaf_config: dict[str, Any]) -> dict[str, Any]:
    benchmark_family = _selection_value(selection_map, "benchmark_family")
    benchmark_config = dict(leaf_config.get("benchmark_config", {}))
    if benchmark_family == "custom_benchmark":
        plugin_path = benchmark_config.get("plugin_path")
        callable_name = benchmark_config.get("callable_name")
        missing = [key for key, value in {"plugin_path": plugin_path, "callable_name": callable_name}.items() if not value]
        if missing:
            raise CompileValidationError(
                f"custom_benchmark requires benchmark_config fields: {missing}"
            )
    return {
        "benchmark_family": benchmark_family,
        **benchmark_config,
    }


def _model_spec(selection_map: dict[str, AxisSelection]) -> dict[str, Any]:
    model_values = selection_map["model_family"].selected_values
    feature_values = selection_map["feature_builder"].selected_values
    return {
        "model_family": model_values[0],
        "feature_builder": feature_values[0],
        "framework": _selection_value(selection_map, "framework"),
        "model_family_values": list(model_values),
        "feature_builder_values": list(feature_values),
    }


def _selection_values(selection: AxisSelection) -> Any:
    if selection.selection_mode == "fixed" and len(selection.selected_values) == 1:
        return selection.selected_values[0]
    return list(selection.selected_values)


def _json_like(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_like(item) for item in value]
    if isinstance(value, list):
        return [_json_like(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_like(item) for key, item in value.items()}
    return value


def _build_tree_context(
    stage0,
    run_spec: RunSpec,
    selections: tuple[AxisSelection, ...],
    leaf_config: dict[str, Any],
) -> dict[str, Any]:
    stage0_payload = design_to_dict(stage0)
    fixed_axes: dict[str, Any] = {}
    sweep_axes: dict[str, Any] = {}
    conditional_axes: dict[str, Any] = {}
    axis_layers: dict[str, str] = {}
    for selection in selections:
        axis_layers[selection.axis_name] = selection.layer
        value = _selection_values(selection)
        if selection.selection_mode == "fixed":
            fixed_axes[selection.axis_name] = value
        elif selection.selection_mode == "sweep":
            sweep_axes[selection.axis_name] = value
        elif selection.selection_mode == "conditional":
            conditional_axes[selection.axis_name] = value
    reproducibility_mode = leaf_config.get("reproducibility_mode_override")
    if reproducibility_mode is None:
        reproducibility_mode = next((selection.selected_values[0] for selection in selections if selection.axis_name == "reproducibility_mode"), "best_effort")
    failure_policy = next((selection.selected_values[0] for selection in selections if selection.axis_name == "failure_policy"), "fail_fast")
    return {
        "research_design": stage0.research_design,
        "design_shape": stage0.design_shape,
        "execution_posture": stage0.execution_posture,
        "experiment_unit": stage0.experiment_unit,
        "reproducibility_mode": reproducibility_mode,
        "failure_policy": failure_policy,
        "compute_mode": next((selection.selected_values[0] for selection in selections if selection.axis_name == "compute_mode"), "serial"),
        "route_owner": resolve_route_owner(stage0),
        "fixed_design": _json_like(stage0_payload["fixed_design"]),
        "varying_design": _json_like(stage0_payload["varying_design"]),
        "comparison_contract": _json_like(stage0_payload["comparison_contract"]),
        "fixed_axes": _json_like(fixed_axes),
        "sweep_axes": _json_like(sweep_axes),
        "conditional_axes": _json_like(conditional_axes),
        "axis_layers": _json_like(axis_layers),
        "leaf_config": _json_like(dict(leaf_config)),
    }


def _tree_context_summary(tree_context: dict[str, Any]) -> str:
    fixed_names = ",".join(sorted(tree_context["fixed_axes"])) or "none"
    sweep_names = ",".join(sorted(tree_context["sweep_axes"])) or "none"
    conditional_names = ",".join(sorted(tree_context["conditional_axes"])) or "none"
    return (
        f"tree_context=route_owner={tree_context['route_owner']}; "
        f"execution_posture={tree_context['execution_posture']}; "
        f"fixed_axes=[{fixed_names}]; "
        f"sweep_axes=[{sweep_names}]; "
        f"conditional_axes=[{conditional_names}]"
    )


def _first_selected_value(selection_map: dict[str, AxisSelection], axis_name: str, default: str) -> str:
    selection = selection_map.get(axis_name)
    if selection is None or not selection.selected_values:
        return default
    return selection.selected_values[0]


def _data_task_spec(selection_map: dict[str, AxisSelection], leaf_config: dict[str, Any]) -> dict[str, Any]:
    dataset = _first_selected_value(selection_map, "dataset", "fred_md")
    task = _first_selected_value(selection_map, "task", "single_target_point_forecast")
    framework = _first_selected_value(selection_map, "framework", "expanding")
    feature_builder = _first_selected_value(selection_map, "feature_builder", "autoreg_lagged_target")
    information_set_type = _first_selected_value(selection_map, "information_set_type", "revised")
    predictor_family_default = "target_lags_only" if feature_builder == "autoreg_lagged_target" else "all_macro_vars"
    return {
        "custom_data_path": leaf_config.get("custom_data_path"),
        "dataset_source": _selection_value(selection_map, "dataset_source", default=dataset),
        "frequency": _selection_value(selection_map, "frequency", default=_DATASET_DEFAULT_FREQUENCY.get(dataset, "monthly")),
        "information_set_type": information_set_type,
        "vintage_policy": _selection_value(selection_map, "vintage_policy", default=("single_vintage" if information_set_type == "real_time_vintage" else "latest_only")),
        "alignment_rule": _selection_value(selection_map, "alignment_rule", default="end_of_period"),
        "forecast_type": _selection_value(selection_map, "forecast_type", default="direct"),
        "forecast_object": _selection_value(selection_map, "forecast_object", default="point_mean"),
        "horizon_target_construction": _selection_value(selection_map, "horizon_target_construction", default="future_level_y_t_plus_h"),
        "overlap_handling": _selection_value(selection_map, "overlap_handling", default="allow_overlap"),
        "predictor_family": _selection_value(selection_map, "predictor_family", default=predictor_family_default),
        "contemporaneous_x_rule": _selection_value(selection_map, "contemporaneous_x_rule", default="forbid_contemporaneous"),
        "own_target_lags": _selection_value(selection_map, "own_target_lags", default="include"),
        "deterministic_components": _selection_value(selection_map, "deterministic_components", default="none"),
        "exogenous_block": _selection_value(selection_map, "exogenous_block", default=("endogenous_allowed" if feature_builder == "raw_feature_panel" else "none")),
        "training_start_rule": _selection_value(selection_map, "training_start_rule", default=("rolling_train_start" if framework == "rolling" else "earliest_possible")),
        "oos_period": _selection_value(selection_map, "oos_period", default=("rolling_origin" if framework == "rolling" else "single_oos_block")),
        "min_train_size": _selection_value(selection_map, "min_train_size", default="fixed_n_obs"),
        "warmup_rule": _selection_value(selection_map, "warmup_rule", default="lags_only_warmup"),
        "structural_break_segmentation": _selection_value(selection_map, "structural_break_segmentation", default="none"),
        "x_map_policy": _selection_value(selection_map, "x_map_policy", default="shared_X"),
        "target_to_target_inclusion": _selection_value(selection_map, "target_to_target_inclusion", default="forbid_other_targets_as_X"),
        "evaluation_scale": _selection_value(selection_map, "evaluation_scale", default="original_scale"),
        "benchmark_family": _selection_value(selection_map, "benchmark_family"),
        "regime_task": _selection_value(selection_map, "regime_task", default="unconditional"),
        "data_vintage": leaf_config.get("data_vintage"),
    }


def _training_spec(selection_map: dict[str, AxisSelection], leaf_config: dict[str, Any]) -> dict[str, Any]:
    framework = _first_selected_value(selection_map, "framework", "expanding")
    feature_builder = _first_selected_value(selection_map, "feature_builder", "autoreg_lagged_target")
    model_family = _first_selected_value(selection_map, "model_family", "ar")
    training_cfg = dict(leaf_config.get("training_config", {}))
    return {
        "outer_window": _selection_value(selection_map, "outer_window", default=framework),
        "refit_policy": _selection_value(selection_map, "refit_policy", default="refit_every_step"),
        "data_richness_mode": _selection_value(selection_map, "data_richness_mode", default=("target_lags_only" if feature_builder == "autoreg_lagged_target" else "full_high_dimensional_X")),
        "sequence_framework": _selection_value(selection_map, "sequence_framework", default="not_sequence"),
        "horizon_modelization": _selection_value(selection_map, "horizon_modelization", default="separate_model_per_h"),
        "validation_size_rule": _selection_value(selection_map, "validation_size_rule", default="ratio"),
        "validation_location": _selection_value(selection_map, "validation_location", default="last_block"),
        "embargo_gap": _selection_value(selection_map, "embargo_gap", default="none"),
        "split_family": _selection_value(selection_map, "split_family", default="time_split"),
        "shuffle_rule": _selection_value(selection_map, "shuffle_rule", default="forbidden_for_time_series"),
        "alignment_fairness": _selection_value(selection_map, "alignment_fairness", default="same_split_across_models"),
        "search_algorithm": _selection_value(selection_map, "search_algorithm", default="grid_search"),
        "tuning_objective": _selection_value(selection_map, "tuning_objective", default="validation_mse"),
        "tuning_budget": _selection_value(selection_map, "tuning_budget", default="max_trials"),
        "hp_space_style": _selection_value(selection_map, "hp_space_style", default="discrete_grid"),
        "seed_policy": _selection_value(selection_map, "seed_policy", default="fixed_seed"),
        "early_stopping": _selection_value(selection_map, "early_stopping", default="none"),
        "convergence_handling": _selection_value(selection_map, "convergence_handling", default="mark_fail"),
        "y_lag_count": _selection_value(selection_map, "y_lag_count", default=("IC_select" if model_family == "ar" else "fixed")),
        "factor_count": _selection_value(selection_map, "factor_count", default="fixed"),
        "lookback": _selection_value(selection_map, "lookback", default="fixed_lookback"),
        "logging_level": _selection_value(selection_map, "logging_level", default="silent"),
        "checkpointing": _selection_value(selection_map, "checkpointing", default="none"),
        "cache_policy": _selection_value(selection_map, "cache_policy", default="no_cache"),
        "execution_backend": _selection_value(selection_map, "execution_backend", default="local_cpu"),
        "forecast_object": _selection_value(selection_map, "forecast_object", default="point_mean"),
        "quantile_level": leaf_config.get("quantile_level", 0.5),
        "validation_ratio": training_cfg.get("validation_ratio", 0.2),
        "validation_n": training_cfg.get("validation_n", 5),
        "validation_years": training_cfg.get("validation_years", 1),
        "obs_per_year": training_cfg.get("obs_per_year", 12),
        "max_trials": training_cfg.get("max_trials", 6),
        "max_time_seconds": training_cfg.get("max_time_seconds", 15.0),
        "early_stop_trials": training_cfg.get("early_stop_trials", 3),
        "early_stop_min_delta": training_cfg.get("early_stop_min_delta", 1e-4),
        "embargo_gap_size": training_cfg.get("embargo_gap_size", 0),
        "fixed_factor_count": training_cfg.get("fixed_factor_count", 3),
        "max_factors": training_cfg.get("max_factors", 5),
        "factor_ar_lags": training_cfg.get("factor_ar_lags", 1),
        "refit_k_steps": training_cfg.get("refit_k_steps", 3),
        "anchored_max_window_size": training_cfg.get("anchored_max_window_size", 60),
        "random_seed": leaf_config.get("random_seed", 42),
    }


def _evaluation_spec(selection_map: dict[str, AxisSelection], leaf_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "primary_metric": _selection_value(selection_map, "primary_metric", default="msfe"),
        "point_metrics": _selection_value(selection_map, "point_metrics", default="MSFE"),
        "relative_metrics": _selection_value(selection_map, "relative_metrics", default="relative_MSFE"),
        "direction_metrics": _selection_value(selection_map, "direction_metrics", default="directional_accuracy"),
        "density_metrics": _selection_value(selection_map, "density_metrics", default="pinball_loss"),
        "economic_metrics": _selection_value(selection_map, "economic_metrics", default="utility_gain"),
        "benchmark_window": _selection_value(selection_map, "benchmark_window", default="expanding"),
        "benchmark_scope": _selection_value(selection_map, "benchmark_scope", default="same_for_all"),
        "agg_time": _selection_value(selection_map, "agg_time", default="full_oos_average"),
        "agg_horizon": _selection_value(selection_map, "agg_horizon", default="equal_weight"),
        "agg_target": _selection_value(selection_map, "agg_target", default="report_separately_only"),
        "ranking": _selection_value(selection_map, "ranking", default="mean_metric_rank"),
        "report_style": _selection_value(selection_map, "report_style", default="tidy_dataframe"),
        "regime_definition": _selection_value(selection_map, "regime_definition", default="none"),
        "regime_use": _selection_value(selection_map, "regime_use", default="eval_only"),
        "regime_metrics": _selection_value(selection_map, "regime_metrics", default="all_main_metrics_by_regime"),
        "decomposition_target": _selection_value(selection_map, "decomposition_target", default="preprocessing_effect"),
        "decomposition_order": _selection_value(selection_map, "decomposition_order", default="marginal_effect_only"),
        "regime_start": leaf_config.get("regime_start"),
        "regime_end": leaf_config.get("regime_end"),
    }


def _build_stage0_and_recipe(
    recipe_dict: dict[str, Any],
    selection_map: dict[str, AxisSelection],
    leaf_config: dict[str, Any],
):
    research_design = _selection_value(selection_map, "research_design")
    dataset = _selection_value(selection_map, "dataset")
    information_set_type = _selection_value(selection_map, "information_set_type")
    task = _selection_value(selection_map, "task")
    benchmark = _selection_value(selection_map, "benchmark_family")
    framework = _selection_value(selection_map, "framework")
    target = leaf_config.get("target", "")
    targets = tuple(leaf_config.get("targets", ()))
    horizons = tuple(leaf_config["horizons"])
    data_vintage = leaf_config.get("data_vintage")
    model_axis = selection_map["model_family"]
    feature_axis = selection_map["feature_builder"]
    feature_builders = feature_axis.selected_values
    wrapper_family = leaf_config.get("wrapper_family")

    if information_set_type == "real_time_vintage" and not data_vintage:
        raise CompileValidationError("information_set_type='real_time_vintage' requires leaf_config.data_vintage")
    if task == "multi_target_point_forecast":
        if len(targets) < 2:
            raise CompileValidationError("task='multi_target_point_forecast' requires leaf_config.targets with at least two entries")
    else:
        if not target:
            raise CompileValidationError("single-target recipes require leaf_config.target")

    derived_experiment_unit = derive_experiment_unit_default(
        research_design=research_design,
        task=task,
        model_axis_mode=model_axis.selection_mode,
        feature_axis_mode=feature_axis.selection_mode,
        wrapper_family=wrapper_family,
    )
    experiment_unit_explicit = "experiment_unit" in selection_map
    experiment_unit = _selection_value(selection_map, "experiment_unit", default=derived_experiment_unit)
    if experiment_unit_explicit:
        unit_entry = get_experiment_unit_entry(experiment_unit)
        if experiment_unit != derived_experiment_unit:
            raise CompileValidationError(
                f"experiment_unit={experiment_unit!r} conflicts with current recipe shape; implied unit is {derived_experiment_unit!r}"
            )
        if unit_entry.requires_multi_target and task != "multi_target_point_forecast":
            raise CompileValidationError(
                f"experiment_unit={experiment_unit!r} requires task='multi_target_point_forecast'"
            )
        if not unit_entry.requires_multi_target and task == "multi_target_point_forecast":
            raise CompileValidationError(
                f"experiment_unit={experiment_unit!r} is incompatible with task='multi_target_point_forecast'"
            )

    sample_split = {
        "expanding": "expanding_window_oos",
        "rolling": "rolling_window_oos",
    }[framework]
    info_set_token = {
        "revised": "revised_monthly",
        "real_time_vintage": "real_time_vintage",
        "pseudo_oos_revised": "pseudo_oos_revised",
        "pseudo_oos_vintage_aware": "pseudo_oos_vintage_aware",
        "release_calendar_aware": "release_calendar_aware",
        "publication_lag_aware": "publication_lag_aware",
    }.get(information_set_type, information_set_type)

    stage0 = build_design_frame(
        research_design=research_design,
        experiment_unit=experiment_unit if experiment_unit_explicit else None,
        fixed_design={
            "dataset_adapter": dataset,
            "information_set": info_set_token,
            "sample_split": sample_split,
            "benchmark": benchmark,
            "evaluation_protocol": "point_forecast_core",
            "forecast_task": task,
        },
        comparison_contract={
            "information_set_policy": "identical",
            "sample_split_policy": "identical",
            "benchmark_policy": "identical",
            "evaluation_policy": "identical",
        },
        varying_design={
            "model_families": model_axis.selected_values,
            "feature_recipes": feature_builders,
            "horizons": tuple(f"h{h}" for h in horizons),
        },
    )
    benchmark_spec = _benchmark_spec(selection_map, leaf_config)
    if framework == "rolling":
        rolling_window_size = int(benchmark_spec.get("rolling_window_size", benchmark_spec.get("minimum_train_size", 5)))
        minimum_train_size = int(benchmark_spec.get("minimum_train_size", 5))
        if rolling_window_size < minimum_train_size:
            raise CompileValidationError("rolling_window_size must be at least minimum_train_size for rolling framework")
    recipe_spec = build_recipe_spec(
        recipe_id=recipe_dict["recipe_id"],
        stage0=stage0,
        target=target,
        horizons=horizons,
        raw_dataset=dataset,
        benchmark_config=benchmark_spec,
        data_task_spec=_data_task_spec(selection_map, leaf_config),
        training_spec=_training_spec(selection_map, leaf_config),
        data_vintage=data_vintage,
        targets=targets,
    )
    run_spec = build_run_spec(recipe_spec)
    return stage0, recipe_spec, run_spec


def _execution_status(
    selections: tuple[AxisSelection, ...],
    preprocess_contract,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    warnings: list[str] = []
    blocked: list[str] = []
    selection_map = _selection_map(selections)
    registry = get_axis_registry()

    failure_policy = _selection_value(selection_map, "failure_policy", default="fail_fast")

    for selection in selections:
        entry = registry[selection.axis_name]
        if selection.selection_mode == "sweep" and entry.default_policy == "fixed":
            warnings.append(
                f"fixed-policy axis '{selection.axis_name}' placed in sweep_axes; grammar allows representation, but governance expects a fixed selection for this axis"
            )
        if selection.selection_mode in {"sweep", "conditional"} and len(selection.selected_values) > 1:
            warnings.append(
                f"axis {selection.axis_name} uses internal sweep values {selection.selected_values}; internal sweep runtime not yet operational"
            )
        for value, status in selection.selected_status.items():
            if status in {"registry_only", "planned", "external_plugin", "not_supported_yet"}:
                warnings.append(
                    f"axis {selection.axis_name} value {value} is representable but not executable (status={status})"
                )

    if not is_operational_preprocess_contract(preprocess_contract):
        warnings.append("preprocessing contract is representable but not executable in the current runtime slice")

    if preprocess_contract.tcode_policy == "raw_only" and preprocess_contract.preprocess_order != "none":
        blocked.append("raw_only tcode_policy cannot be paired with non-none preprocess_order")

    model_family = _selection_value(selection_map, "model_family") if "model_family" in selection_map and len(selection_map["model_family"].selected_values) == 1 else None
    feature_builder = _selection_value(selection_map, "feature_builder") if "feature_builder" in selection_map and len(selection_map["feature_builder"].selected_values) == 1 else None
    if model_family == "ar" and feature_builder == "raw_feature_panel":
        blocked.append("raw_feature_panel is not compatible with model_family='ar' in the current runtime slice")
    forecast_object = _selection_value(selection_map, "forecast_object", default="point_mean")
    if model_family == "quantile_linear" and forecast_object != "point_median":
        blocked.append("model_family='quantile_linear' currently requires forecast_object='point_median'")

    if feature_builder is not None:
        predictor_family = _selection_value(selection_map, "predictor_family", default=("target_lags_only" if feature_builder == "autoreg_lagged_target" else "all_macro_vars"))
        if predictor_family == "target_lags_only" and feature_builder != "autoreg_lagged_target":
            blocked.append("predictor_family='target_lags_only' requires feature_builder='autoreg_lagged_target' in the current runtime slice")
        if predictor_family == "all_macro_vars" and feature_builder not in {"raw_feature_panel", "factor_pca", "factors_plus_AR"}:
            blocked.append("predictor_family='all_macro_vars' requires feature_builder in {'raw_feature_panel', 'factor_pca', 'factors_plus_AR'} in the current runtime slice")

    if failure_policy not in {"fail_fast", "skip_failed_cell", "skip_failed_model", "save_partial_results", "warn_only"}:
        warnings.append(
            f"failure_policy {failure_policy!r} is representable but not executable in the current runtime slice"
        )
    research_design = _selection_value(selection_map, "research_design", default="single_path_benchmark")
    if research_design in {"orchestrated_bundle"}:
        warnings.append(
            f"research_design={research_design!r} uses the wrapper/orchestrator route; execute via a wrapper runtime rather than single-path execute_recipe"
        )
    experiment_unit = _selection_value(selection_map, "experiment_unit", default="single_target_single_model") if "experiment_unit" in selection_map else "single_target_single_model"
    if experiment_unit in {"benchmark_suite"}:
        warnings.append(
            f"experiment_unit={experiment_unit!r} is a wrapper-managed unit; execute via the wrapper runtime"
        )
    compute_mode = _selection_value(selection_map, "compute_mode", default="serial")
    if compute_mode not in {"serial", "parallel_by_model", "parallel_by_horizon", "parallel_by_target", "parallel_by_oos_date"}:
        warnings.append(
            f"compute_mode {compute_mode!r} is representable but not executable in the current runtime slice"
        )

    if blocked:
        return "blocked_by_incompatibility", tuple(warnings), tuple(blocked)

    executable = not warnings
    if executable:
        return "executable", (), ()
    return "representable_but_not_executable", tuple(warnings), ()


def _build_wrapper_handoff(
    stage0,
    recipe_spec: RecipeSpec,
    run_spec: RunSpec,
    leaf_config: dict[str, Any],
    *,
    experiment_unit_explicit: bool,
) -> dict[str, Any]:
    if run_spec.route_owner != "wrapper":
        return {}
    wrapper_family = leaf_config.get("wrapper_family")
    bundle_label = leaf_config.get("bundle_label")
    if experiment_unit_explicit:
        wrapper_family = wrapper_family or stage0.experiment_unit
        bundle_label = bundle_label or f"{recipe_spec.recipe_id}-{wrapper_family}"
    if wrapper_family not in {
        "single_target_full_sweep",
        "multi_target_separate_runs",
        "multi_target_shared_design",
        "benchmark_suite",
        "ablation_study",
    }:
        raise CompileValidationError(
            "wrapper_bundle_plan requires a wrapper family in {'single_target_full_sweep', 'multi_target_separate_runs', 'multi_target_shared_design', 'benchmark_suite', 'ablation_study'}"
        )
    if not isinstance(bundle_label, str) or not bundle_label.strip():
        raise CompileValidationError("wrapper_bundle_plan requires non-empty leaf_config.bundle_label")
    return {
        "wrapper_family": wrapper_family,
        "bundle_label": bundle_label,
        "route_owner": run_spec.route_owner,
        "execution_posture": stage0.execution_posture,
        "experiment_unit": stage0.experiment_unit,
        "recipe_id": recipe_spec.recipe_id,
        "artifact_subdir": run_spec.artifact_subdir,
        "targets": list(recipe_spec.targets),
        "horizons": list(recipe_spec.horizons),
    }


def compile_recipe_dict(recipe_dict: dict[str, Any]) -> CompileResult:
    from macrocast.compiler.migrations import migrate_legacy_stat_test
    recipe_dict = migrate_legacy_stat_test(recipe_dict)
    if not recipe_dict.get("recipe_id"):
        raise CompileValidationError("recipe_id is required")
    selections = _build_axis_selections(recipe_dict)
    _ensure_unique_axis_selections(selections)
    selection_map = _selection_map(selections)
    required_axes = {"research_design", "dataset", "information_set_type", "task", "framework", "benchmark_family", "model_family", "feature_builder"}
    missing_axes = sorted(axis for axis in required_axes if axis not in selection_map)
    if missing_axes:
        raise CompileValidationError(f"recipe missing required axes: {missing_axes}")
    leaf_config = _leaf_config(recipe_dict)
    if "horizons" not in leaf_config:
        raise CompileValidationError("recipe leaf_config missing 'horizons'")
    # Resolve declarative derived_axes (axis_type=derived). Each derivation rule
    # computes a concrete value from the current selection_map + leaf_config and
    # is appended to the selection tuple with selection_mode="derived".
    derived_additions = _resolve_derived_axes(recipe_dict, selection_map, leaf_config)
    if derived_additions:
        selections = selections + tuple(derived_additions)
        _ensure_unique_axis_selections(selections)
        selection_map = _selection_map(selections)
    task_value = _selection_value(selection_map, "task")
    experiment_unit_explicit = "experiment_unit" in selection_map
    if task_value == "multi_target_point_forecast":
        if "targets" not in leaf_config:
            raise CompileValidationError("recipe leaf_config missing 'targets'")
    else:
        if "target" not in leaf_config:
            raise CompileValidationError("recipe leaf_config missing 'target'")

    # Custom data source validation: custom_csv / custom_parquet require leaf_config.custom_data_path
    ds_source_choice = selection_map["dataset_source"].selected_values[0] if "dataset_source" in selection_map else None
    if ds_source_choice in {"custom_csv", "custom_parquet"} and not leaf_config.get("custom_data_path"):
        raise CompileValidationError(
            f"dataset_source={ds_source_choice!r} requires leaf_config.custom_data_path"
        )
    reproducibility_mode = _selection_value(selection_map, "reproducibility_mode", default="best_effort")
    random_seed = leaf_config.get("random_seed")
    if reproducibility_mode in {"strict_reproducible", "seeded_reproducible"} and random_seed is None:
        raise CompileValidationError(
            f"reproducibility_mode={reproducibility_mode!r} requires leaf_config.random_seed"
        )
    failure_policy = _selection_value(selection_map, "failure_policy", default="fail_fast")
    compute_mode = _selection_value(selection_map, "compute_mode", default="serial")

    preprocess_contract = _build_preprocess_contract(selection_map)
    stage0, recipe_spec, run_spec = _build_stage0_and_recipe(recipe_dict, selection_map, leaf_config)
    execution_status, warnings, blocked = _execution_status(selections, preprocess_contract)
    tree_context = _build_tree_context(stage0, run_spec, selections, leaf_config)
    wrapper_handoff = _build_wrapper_handoff(
        stage0,
        recipe_spec,
        run_spec,
        leaf_config,
        experiment_unit_explicit=experiment_unit_explicit,
    )

    compiled = CompiledRecipeSpec(
        recipe_id=recipe_dict["recipe_id"],
        layer_order=get_canonical_layer_order(),
        axis_selections=selections,
        leaf_config=leaf_config,
        preprocess_contract=preprocess_contract,
        stage0=stage0,
        recipe_spec=recipe_spec,
        run_spec=run_spec,
        execution_status=execution_status,
        warnings=warnings,
        blocked_reasons=blocked,
        tree_context=tree_context,
        wrapper_handoff=wrapper_handoff,
    )
    manifest = compiled_spec_to_dict(compiled)
    return CompileResult(compiled=compiled, manifest=manifest)


def compile_recipe_yaml(path: str | Path) -> CompileResult:
    return compile_recipe_dict(load_recipe_yaml(path))



def _output_spec(selection_map):
    return {
        "export_format": _selection_value(selection_map, "export_format", default="json"),
        "saved_objects": _selection_value(selection_map, "saved_objects", default="full_bundle"),
        "provenance_fields": _selection_value(selection_map, "provenance_fields", default="full"),
        "artifact_granularity": _selection_value(selection_map, "artifact_granularity", default="aggregated"),
    }
def compiled_spec_to_dict(compiled: CompiledRecipeSpec) -> dict[str, Any]:
    selection_map = {selection.axis_name: selection for selection in compiled.axis_selections}
    return {
        "recipe_id": compiled.recipe_id,
        "layer_order": list(compiled.layer_order),
        "execution_status": compiled.execution_status,
        "warnings": list(compiled.warnings),
        "blocked_reasons": list(compiled.blocked_reasons),
        "leaf_config": dict(compiled.leaf_config),
        "benchmark_spec": dict(compiled.recipe_spec.benchmark_config),
        "model_spec": _model_spec(selection_map),
        "reproducibility_spec": {
            "reproducibility_mode": _selection_value(selection_map, "reproducibility_mode", default="best_effort"),
            "random_seed": compiled.leaf_config.get("random_seed"),
        },
        "failure_policy_spec": {
            "failure_policy": _selection_value(selection_map, "failure_policy", default="fail_fast"),
        },
        "compute_mode_spec": {
            "compute_mode": _selection_value(selection_map, "compute_mode", default="serial"),
        },
        "data_task_spec": _data_task_spec(selection_map, compiled.leaf_config),
        "training_spec": _training_spec(selection_map, compiled.leaf_config),
        "evaluation_spec": _evaluation_spec(selection_map, compiled.leaf_config),
        "stat_test_spec": {
            "stat_test": _selection_value(selection_map, "stat_test", default="none"),
            "dependence_correction": _selection_value(selection_map, "dependence_correction", default="none"),
        },
        "importance_spec": {
            "importance_method": _selection_value(selection_map, "importance_method"),
        },
        "output_spec": _output_spec(selection_map),
        "preprocess_contract": preprocess_to_dict(compiled.preprocess_contract),
        "axis_selections": [
            {
                "axis_name": selection.axis_name,
                "layer": selection.layer,
                "selection_mode": selection.selection_mode,
                "selected_values": list(selection.selected_values),
                "selected_status": dict(selection.selected_status),
            }
            for selection in compiled.axis_selections
        ],
        "run_spec": {
            "run_id": compiled.run_spec.run_id,
            "artifact_subdir": compiled.run_spec.artifact_subdir,
            "route_owner": compiled.run_spec.route_owner,
        },
        "tree_context": dict(compiled.tree_context),
        "wrapper_handoff": dict(compiled.wrapper_handoff),
    }


def run_compiled_recipe(
    compiled: CompiledRecipeSpec,
    *,
    output_root: str | Path,
    local_raw_source: str | Path | None = None,
):
    if compiled.execution_status != "executable":
        raise CompileValidationError(
            f"compiled recipe is not executable: {compiled.execution_status}; warnings={compiled.warnings}; blocked={compiled.blocked_reasons}"
        )
    return execute_recipe(
        recipe=compiled.recipe_spec,
        preprocess=compiled.preprocess_contract,
        output_root=output_root,
        local_raw_source=local_raw_source,
        provenance_payload={"compiler": compiled_spec_to_dict(compiled), "tree_context": dict(compiled.tree_context)},
    )
