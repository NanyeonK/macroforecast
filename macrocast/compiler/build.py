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
from ..stage0 import build_stage0_frame, resolve_route_owner, stage0_to_dict

_ALLOWED_SELECTION_MODES = ("fixed_axes", "sweep_axes", "conditional_axes", "leaf_config")


def load_recipe_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _normalize_layer_spec(layer_spec: dict[str, Any] | None) -> dict[str, Any]:
    layer_spec = layer_spec or {}
    return {key: dict(layer_spec.get(key, {})) for key in _ALLOWED_SELECTION_MODES}


def _build_axis_selections(recipe_dict: dict[str, Any]) -> tuple[AxisSelection, ...]:
    registry = get_axis_registry()
    path = recipe_dict.get("path", {})
    selections: list[AxisSelection] = []
    for layer in get_canonical_layer_order():
        layer_spec = _normalize_layer_spec(path.get(layer))
        for selection_mode in ("fixed_axes", "sweep_axes", "conditional_axes"):
            mode_name = selection_mode.replace("_axes", "")
            for axis_name, raw_value in layer_spec[selection_mode].items():
                if axis_name not in registry:
                    raise CompileValidationError(f"unknown registry axis {axis_name!r}")
                entry = registry[axis_name]
                values = tuple(raw_value) if isinstance(raw_value, list) else (raw_value,)
                for value in values:
                    if value not in entry.allowed_values:
                        raise CompileValidationError(
                            f"axis {axis_name!r} received unknown value {value!r}"
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


def _leaf_config(recipe_dict: dict[str, Any]) -> dict[str, Any]:
    path = recipe_dict.get("path", {})
    leaf: dict[str, Any] = {}
    for layer in get_canonical_layer_order():
        leaf.update(_normalize_layer_spec(path.get(layer))["leaf_config"])
    return leaf


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
    missing = sorted(axis for axis in required if axis not in selection_map)
    if missing:
        raise CompileValidationError(f"preprocessing layer missing required axes: {missing}")
    try:
        contract = build_preprocess_contract(
            **{axis: selection_map[axis].selected_values[0] for axis in required}
        )
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
    stage0_payload = stage0_to_dict(stage0)
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
        "study_mode": stage0.study_mode,
        "design_shape": stage0.design_shape,
        "execution_posture": stage0.execution_posture,
        "experiment_unit": stage0.experiment_unit,
        "reproducibility_mode": reproducibility_mode,
        "failure_policy": failure_policy,
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


def _build_stage0_and_recipe(
    recipe_dict: dict[str, Any],
    selection_map: dict[str, AxisSelection],
    leaf_config: dict[str, Any],
):
    study_mode = _selection_value(selection_map, "study_mode")
    dataset = _selection_value(selection_map, "dataset")
    info_set = _selection_value(selection_map, "info_set")
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

    if info_set == "real_time" and not data_vintage:
        raise CompileValidationError("info_set='real_time' requires leaf_config.data_vintage")
    if task == "multi_target_point_forecast":
        if len(targets) < 2:
            raise CompileValidationError("task='multi_target_point_forecast' requires leaf_config.targets with at least two entries")
    else:
        if not target:
            raise CompileValidationError("single-target recipes require leaf_config.target")

    derived_experiment_unit = derive_experiment_unit_default(
        study_mode=study_mode,
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
        "real_time": "real_time_vintage",
    }[info_set]

    stage0 = build_stage0_frame(
        study_mode=study_mode,
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

    if failure_policy not in {"fail_fast", "hard_error", "skip_failed_model", "save_partial_results"}:
        warnings.append(
            f"failure_policy {failure_policy!r} is representable but not executable in the current runtime slice"
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
    if not recipe_dict.get("recipe_id"):
        raise CompileValidationError("recipe_id is required")
    selections = _build_axis_selections(recipe_dict)
    selection_map = _selection_map(selections)
    required_axes = {"study_mode", "dataset", "info_set", "task", "framework", "benchmark_family", "model_family", "feature_builder"}
    missing_axes = sorted(axis for axis in required_axes if axis not in selection_map)
    if missing_axes:
        raise CompileValidationError(f"recipe missing required axes: {missing_axes}")
    leaf_config = _leaf_config(recipe_dict)
    if "horizons" not in leaf_config:
        raise CompileValidationError("recipe leaf_config missing 'horizons'")
    task_value = _selection_value(selection_map, "task")
    experiment_unit_explicit = "experiment_unit" in selection_map
    if task_value == "multi_target_point_forecast":
        if "targets" not in leaf_config:
            raise CompileValidationError("recipe leaf_config missing 'targets'")
    else:
        if "target" not in leaf_config:
            raise CompileValidationError("recipe leaf_config missing 'target'")

    reproducibility_mode = _selection_value(selection_map, "reproducibility_mode", default="best_effort")
    random_seed = leaf_config.get("random_seed")
    if reproducibility_mode in {"strict_reproducible", "seeded_reproducible"} and random_seed is None:
        raise CompileValidationError(
            f"reproducibility_mode={reproducibility_mode!r} requires leaf_config.random_seed"
        )
    failure_policy = _selection_value(selection_map, "failure_policy", default="fail_fast")

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
        "stat_test_spec": {
            "stat_test": _selection_value(selection_map, "stat_test"),
        },
        "importance_spec": {
            "importance_method": _selection_value(selection_map, "importance_method"),
        },
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
