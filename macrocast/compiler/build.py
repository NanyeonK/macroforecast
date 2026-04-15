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
from ..stage0 import build_stage0_frame

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
    target = leaf_config["target"]
    horizons = tuple(leaf_config["horizons"])
    model_axis = selection_map["model_family"]
    feature_builders = selection_map["feature_builder"].selected_values

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

    for selection in selections:
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

    if blocked:
        return "blocked_by_incompatibility", tuple(warnings), tuple(blocked)

    executable = not warnings
    if executable:
        return "executable", (), ()
    return "representable_but_not_executable", tuple(warnings), ()


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
    for key in ("target", "horizons"):
        if key not in leaf_config:
            raise CompileValidationError(f"recipe leaf_config missing {key!r}")

    preprocess_contract = _build_preprocess_contract(selection_map)
    stage0, recipe_spec, run_spec = _build_stage0_and_recipe(recipe_dict, selection_map, leaf_config)
    execution_status, warnings, blocked = _execution_status(selections, preprocess_contract)

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
        provenance_payload={"compiler": compiled_spec_to_dict(compiled)},
    )
