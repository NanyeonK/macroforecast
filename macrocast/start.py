from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml

from .compiler import CompileValidationError, compile_recipe_dict, compile_recipe_yaml, load_recipe_yaml
from .execution.importance_dispatch import IMPORTANCE_FILE_NAMES, active_importance_methods
from .execution.stat_tests import active_stat_test_axes
from .registry import get_axis_registry_entry
from .registry.stage0.experiment_unit import derive_experiment_unit_default, experiment_unit_options_for_wizard

_AVAILABLE_STAGES = (
    "route_preview",
    "compile_preview",
    "tree_context",
    "runs_preview",
    "manifest_preview",
)

_WIZARD_KEYS = (
    "research_design",
    "target_structure",
    "experiment_unit",
    "target",
    "targets",
    "framework",
    "benchmark_family",
    "benchmark_plugin_path",
    "benchmark_callable_name",
    "tcode_policy",
    "x_missing_policy",
    "scaling_policy",
    "preprocess_order",
    "preprocess_fit_scope",
    "model_path_mode",
    "model_family",
    "feature_builder",
    "primary_metric",
    "manifest_mode",
    "equal_predictive",
    "importance_method",
)


def _normalize_stages(stages: str | Iterable[str] | None) -> list[str]:
    if stages is None:
        return list(_AVAILABLE_STAGES)
    if isinstance(stages, str):
        stages = [stages]
    normalized: list[str] = []
    for stage in stages:
        if stage not in _AVAILABLE_STAGES:
            raise ValueError(f"unknown stage {stage!r}; valid stages: {_AVAILABLE_STAGES}")
        normalized.append(stage)
    return normalized


def _tree_context_summary(tree_context: dict[str, Any]) -> str:
    fixed_names = ",".join(sorted(tree_context.get("fixed_axes", {}))) or "none"
    sweep_names = ",".join(sorted(tree_context.get("sweep_axes", {}))) or "none"
    conditional_names = ",".join(sorted(tree_context.get("conditional_axes", {}))) or "none"
    return (
        f"route_owner={tree_context.get('route_owner', 'unknown')}; "
        f"execution_posture={tree_context.get('execution_posture', 'unknown')}; "
        f"fixed_axes=[{fixed_names}]; "
        f"sweep_axes=[{sweep_names}]; "
        f"conditional_axes=[{conditional_names}]"
    )


def _recipe_leaf(recipe: dict[str, Any], layer: str) -> dict[str, Any]:
    return recipe.setdefault("path", {}).setdefault(layer, {}).setdefault("leaf_config", {})


def _recipe_fixed(recipe: dict[str, Any], layer: str) -> dict[str, Any]:
    return recipe.setdefault("path", {}).setdefault(layer, {}).setdefault("fixed_axes", {})


def _recipe_sweep(recipe: dict[str, Any], layer: str) -> dict[str, Any]:
    return recipe.setdefault("path", {}).setdefault(layer, {}).setdefault("sweep_axes", {})


def _benchmark_config(recipe: dict[str, Any]) -> dict[str, Any]:
    return _recipe_leaf(recipe, "5_output_provenance").setdefault("benchmark_config", {})


def _choice_option_details(axis_name: str, options: list[str]) -> dict[str, dict[str, str]]:
    try:
        entry = get_axis_registry_entry(axis_name)
    except Exception:
        return {}
    details: dict[str, dict[str, str]] = {}
    for option in options:
        details[option] = {"status": entry.current_status.get(option, "unknown")}
    return details


def _target_structure_value(recipe: dict[str, Any]) -> str:
    fixed = _recipe_fixed(recipe, "1_data_task")
    return fixed.get("target_structure", "single_target")


def _set_target_structure(recipe: dict[str, Any], value: str) -> None:
    fixed = _recipe_fixed(recipe, "1_data_task")
    fixed["target_structure"] = value


def _planned_branch_message(warnings: list[str]) -> str | None:
    planned_hits = [warning for warning in warnings if "status=planned" in warning]
    if not planned_hits:
        return None
    return "Planned branch selected. Current YAML parses, but this branch is not executable in the current single-run runtime."


def _single_run_extension_message(tree_context: dict[str, Any], warnings: list[str]) -> str:
    sweep_axes = tree_context.get("sweep_axes", {})
    if "model_family" in sweep_axes and "feature_builder" in sweep_axes:
        return "Full model/feature sweep is not a single-run preview. Use the sweep runtime only if the compiler reports ready_for_sweep_runner; otherwise drop this route."
    if "model_family" in sweep_axes:
        return "Model grid selected. Execute with compile_sweep_plan/execute_sweep; single-run previews stop at the parent recipe."
    return _planned_branch_message(warnings) or "Sweep route selected. Execute with compile_sweep_plan/execute_sweep; single-run previews stop at the parent recipe."


def _read_wizard_value(recipe: dict[str, Any], key: str) -> Any:
    if key == "research_design":
        return _recipe_fixed(recipe, "0_meta").get("research_design")
    if key == "target_structure":
        return _target_structure_value(recipe)
    if key == "experiment_unit":
        explicit = _recipe_fixed(recipe, "0_meta").get("experiment_unit")
        if explicit:
            return explicit
        training = _recipe_fixed(recipe, "3_training")
        training_sweep = _recipe_sweep(recipe, "3_training")
        return derive_experiment_unit_default(
            research_design=_recipe_fixed(recipe, "0_meta").get("research_design", "single_forecast_run"),
            task=_target_structure_value(recipe),
            model_axis_mode="sweep" if "model_family" in training_sweep else "fixed",
            feature_axis_mode="sweep" if "feature_builder" in training_sweep else "fixed",
            wrapper_family=_recipe_leaf(recipe, "5_output_provenance").get("wrapper_family"),
        )
    if key in {"target", "targets"}:
        return _recipe_leaf(recipe, "1_data_task").get(key)
    if key == "model_path_mode":
        fixed = _recipe_fixed(recipe, "3_training")
        sweep = _recipe_sweep(recipe, "3_training")
        if "model_family" in sweep and "feature_builder" in sweep:
            return "full_sweep"
        if "model_family" in sweep:
            return "model_grid"
        return "single_model"
    if key in {"framework", "benchmark_family", "model_family", "feature_builder"}:
        return _recipe_fixed(recipe, "3_training").get(key)
    if key in {"tcode_policy", "x_missing_policy", "scaling_policy", "preprocess_order", "preprocess_fit_scope"}:
        return _recipe_fixed(recipe, "2_preprocessing").get(key)
    if key == "primary_metric":
        return _recipe_fixed(recipe, "4_evaluation").get(key)
    if key == "manifest_mode":
        return _recipe_leaf(recipe, "5_output_provenance").get(key)
    if key in {"equal_predictive", "dependence_correction"}:
        return _recipe_fixed(recipe, "6_stat_tests").get(key, "none")
    if key == "importance_method":
        return _recipe_fixed(recipe, "7_importance").get(key)
    if key == "benchmark_plugin_path":
        return _benchmark_config(recipe).get("plugin_path")
    if key == "benchmark_callable_name":
        return _benchmark_config(recipe).get("callable_name")
    raise KeyError(key)


def _apply_wizard_value(recipe: dict[str, Any], key: str, value: Any) -> None:
    if key == "research_design":
        _recipe_fixed(recipe, "0_meta")["research_design"] = value
        return
    leaf = _recipe_leaf(recipe, "1_data_task")
    preprocess = _recipe_fixed(recipe, "2_preprocessing")
    training = _recipe_fixed(recipe, "3_training")
    training_sweep = _recipe_sweep(recipe, "3_training")
    if key == "target_structure":
        _set_target_structure(recipe, str(value))
        if value == "multi_target":
            leaf.pop("target", None)
            leaf.setdefault("targets", [])
        else:
            if "targets" in leaf:
                targets = list(leaf.get("targets", []))
                leaf.pop("targets", None)
                if targets:
                    leaf["target"] = targets[0]
            leaf.setdefault("target", "INDPRO")
        return
    if key == "experiment_unit":
        meta = _recipe_fixed(recipe, "0_meta")
        meta["experiment_unit"] = value
        output_leaf = _recipe_leaf(recipe, "5_output_provenance")
        if value == "replication_recipe":
            meta["research_design"] = "replication_recipe"
            output_leaf.pop("wrapper_family", None)
            output_leaf.pop("bundle_label", None)
            return
        if value in {"benchmark_suite", "ablation_study"}:
            meta["research_design"] = "study_bundle"
            _set_target_structure(recipe, "single_target")
            leaf.pop("targets", None)
            leaf.setdefault("target", "INDPRO")
            output_leaf["wrapper_family"] = value
            output_leaf.setdefault("bundle_label", value.replace("_", "-"))
            return
        if value in {"multi_target_separate_runs", "multi_target_shared_design"}:
            meta["research_design"] = "study_bundle"
            _set_target_structure(recipe, "multi_target")
            leaf.pop("target", None)
            leaf.setdefault("targets", ["INDPRO", "RPI"])
            output_leaf["wrapper_family"] = value
            output_leaf.setdefault("bundle_label", value.replace("_", "-"))
            return
        meta["research_design"] = "single_forecast_run"
        _set_target_structure(recipe, "single_target")
        leaf.pop("targets", None)
        leaf.setdefault("target", "INDPRO")
        output_leaf.pop("wrapper_family", None)
        output_leaf.pop("bundle_label", None)
        if value == "single_target_single_generator":
            training["model_family"] = training.get("model_family", "ar")
            training["feature_builder"] = training.get("feature_builder", "target_lag_features")
            training_sweep.pop("model_family", None)
            training_sweep.pop("feature_builder", None)
        elif value == "single_target_generator_grid":
            training["feature_builder"] = training.get("feature_builder", "target_lag_features")
            training.pop("model_family", None)
            training_sweep["model_family"] = ["ar", "ridge", "lasso", "random_forest"]
            training_sweep.pop("feature_builder", None)
        elif value == "single_target_full_sweep":
            output_leaf["wrapper_family"] = value
            output_leaf.setdefault("bundle_label", value.replace("_", "-"))
            training.pop("model_family", None)
            training.pop("feature_builder", None)
            training_sweep["model_family"] = ["ar", "ridge", "lasso", "random_forest"]
            training_sweep["feature_builder"] = ["target_lag_features", "raw_feature_panel", "pca_factor_features"]
        return
    if key == "target":
        leaf["target"] = str(value)
        leaf.pop("targets", None)
        return
    if key == "targets":
        targets = [item.strip() for item in str(value).split(",") if item.strip()]
        leaf["targets"] = targets
        leaf.pop("target", None)
        return
    if key == "model_path_mode":
        current_model = training.get("model_family", "ar")
        current_feature = training.get("feature_builder", "target_lag_features")
        if value == "single_model":
            training["model_family"] = current_model
            training["feature_builder"] = current_feature
            training_sweep.pop("model_family", None)
            training_sweep.pop("feature_builder", None)
            _recipe_fixed(recipe, "0_meta")["experiment_unit"] = "single_target_single_generator"
        elif value == "model_grid":
            training["feature_builder"] = current_feature
            training.pop("model_family", None)
            training_sweep["model_family"] = ["ar", "ridge", "lasso", "random_forest"]
            training_sweep.pop("feature_builder", None)
            _recipe_fixed(recipe, "0_meta")["experiment_unit"] = "single_target_generator_grid"
        elif value == "full_sweep":
            training.pop("model_family", None)
            training.pop("feature_builder", None)
            training_sweep["model_family"] = ["ar", "ridge", "lasso", "random_forest"]
            training_sweep["feature_builder"] = ["target_lag_features", "raw_feature_panel", "pca_factor_features"]
            _recipe_fixed(recipe, "0_meta")["experiment_unit"] = "single_target_full_sweep"
            _recipe_leaf(recipe, "5_output_provenance")["wrapper_family"] = "single_target_full_sweep"
            _recipe_leaf(recipe, "5_output_provenance").setdefault("bundle_label", "single-target-full-sweep")
        return
    if key in {"framework", "benchmark_family", "model_family", "feature_builder"}:
        training[key] = value
        if key in {"model_family", "feature_builder"}:
            training_sweep.pop(key, None)
        if key == "benchmark_family" and value != "custom_benchmark":
            cfg = _benchmark_config(recipe)
            cfg.pop("plugin_path", None)
            cfg.pop("callable_name", None)
        return
    if key == "primary_metric":
        _recipe_fixed(recipe, "4_evaluation")[key] = value
        return
    if key == "manifest_mode":
        _recipe_leaf(recipe, "5_output_provenance")[key] = value
        return
    if key in {"equal_predictive", "dependence_correction"}:
        _recipe_fixed(recipe, "6_stat_tests")[key] = value
        return
    if key == "importance_method":
        _recipe_fixed(recipe, "7_importance")[key] = value
        return
    if key in {"tcode_policy", "x_missing_policy", "scaling_policy", "preprocess_order", "preprocess_fit_scope"}:
        preprocess[key] = value
        if key == "tcode_policy":
            if value == "raw_only":
                preprocess["x_missing_policy"] = "none"
                preprocess["scaling_policy"] = "none"
                preprocess["preprocess_order"] = "none"
                preprocess["preprocess_fit_scope"] = "not_applicable"
            elif value == "extra_preprocess_only":
                preprocess.setdefault("x_missing_policy", "em_impute")
                preprocess.setdefault("scaling_policy", "standard")
                preprocess["preprocess_order"] = "extra_only"
                preprocess["preprocess_fit_scope"] = "train_only"
        return
    if key == "benchmark_plugin_path":
        _benchmark_config(recipe)["plugin_path"] = str(value)
        return
    if key == "benchmark_callable_name":
        _benchmark_config(recipe)["callable_name"] = str(value)
        return
    raise KeyError(key)


def _wizard_choice_stack(recipe: dict[str, Any]) -> list[dict[str, Any]]:
    target_structure = _target_structure_value(recipe)
    target_key = "targets" if target_structure == "multi_target" else "target"
    benchmark_family = _recipe_fixed(recipe, "3_training").get("benchmark_family", "zero_change")
    stack = [
        {
            "key": "research_design",
            "prompt": "Study mode",
            "options": [
                "single_forecast_run",
                "controlled_variation",
            ],
        },
        {
            "key": "target_structure",
            "prompt": "Target structure",
            "options": [
                "single_target",
                "multi_target",
            ],
        },
        {
            "key": "experiment_unit",
            "prompt": "Experiment unit",
            "options": list(experiment_unit_options_for_wizard(
                _recipe_fixed(recipe, "0_meta").get("research_design", "single_forecast_run"),
                target_structure,
            )),
        },
        {
            "key": target_key,
            "prompt": "Targets" if target_key == "targets" else "Target",
            "options": [] if target_key == "targets" else ["INDPRO", "RPI", "UNRATE"],
        },
        {
            "key": "framework",
            "prompt": "Framework",
            "options": ["expanding", "rolling"],
        },
        {
            "key": "benchmark_family",
            "prompt": "Benchmark family",
            "options": ["zero_change", "autoregressive_bic", "historical_mean", "custom_benchmark"],
        },
    ]
    if benchmark_family == "custom_benchmark":
        stack.extend([
            {"key": "benchmark_plugin_path", "prompt": "Benchmark plugin path", "options": []},
            {"key": "benchmark_callable_name", "prompt": "Benchmark callable name", "options": []},
        ])
    stack.extend([
        {
            "key": "tcode_policy",
            "prompt": "T-code policy",
            "options": ["raw_only", "extra_preprocess_only"],
        },
        {
            "key": "x_missing_policy",
            "prompt": "X missing policy",
            "options": ["none", "em_impute"],
        },
        {
            "key": "scaling_policy",
            "prompt": "Scaling policy",
            "options": ["none", "standard", "robust"],
        },
        {
            "key": "preprocess_order",
            "prompt": "Preprocess order",
            "options": ["none", "extra_only"],
        },
        {
            "key": "preprocess_fit_scope",
            "prompt": "Preprocess fit scope",
            "options": ["not_applicable", "train_only"],
        },
        {
            "key": "model_path_mode",
            "prompt": "Model path mode",
            "options": ["single_model", "model_grid"],
        },
        {
            "key": "model_family",
            "prompt": "Model family",
            "options": ["ar", "ridge", "lasso", "random_forest"],
        },
        {
            "key": "feature_builder",
            "prompt": "Feature builder",
            "options": ["target_lag_features", "raw_feature_panel", "pca_factor_features"],
        },
        {
            "key": "primary_metric",
            "prompt": "Primary metric",
            "options": ["msfe", "relative_msfe", "oos_r2", "csfe"],
        },
        {
            "key": "manifest_mode",
            "prompt": "Manifest mode",
            "options": ["full"],
        },
        {
            "key": "equal_predictive",
            "prompt": "Equal predictive test",
            "options": ["none", "dm", "dm_hln", "dm_modified"],
        },
        {
            "key": "importance_method",
            "prompt": "Importance method",
            "options": ["none", "minimal_importance", "tree_shap", "kernel_shap", "linear_shap", "permutation_importance", "lime", "feature_ablation", "pdp", "ice", "ale", "grouped_permutation", "importance_stability"],
        },
    ])
    filtered_stack = []
    model_path_mode = _read_wizard_value(recipe, "model_path_mode")
    for choice in stack:
        if model_path_mode != "single_model" and choice["key"] in {"model_family", "feature_builder"}:
            continue
        if choice["options"]:
            axis_name = choice["key"]
            if axis_name not in {"benchmark_plugin_path", "benchmark_callable_name", "manifest_mode", "model_path_mode"}:
                choice["option_details"] = _choice_option_details(axis_name, list(choice["options"]))
        filtered_stack.append(choice)
    return filtered_stack


def _route_preview(compile_manifest: dict[str, Any]) -> dict[str, Any]:
    tree_context = dict(compile_manifest.get("tree_context", {}))
    route_owner = tree_context.get("route_owner", compile_manifest.get("run_spec", {}).get("route_owner", "unknown"))
    execution_status = compile_manifest.get("execution_status", "unknown")
    warnings = list(compile_manifest.get("warnings", []))
    blocked_reasons = list(compile_manifest.get("blocked_reasons", []))

    if execution_status == "executable":
        wizard_status = "implemented"
        continue_in_single_run = True
        message = "Route remains inside the current executable single-run surface."
    elif execution_status == "ready_for_sweep_runner":
        wizard_status = "sweep_runner_ready"
        continue_in_single_run = False
        message = _single_run_extension_message(tree_context, warnings)
    elif execution_status == "ready_for_wrapper_runner":
        wizard_status = "wrapper_runner_ready"
        continue_in_single_run = False
        message = "Route is wrapper-owned and has a runner contract. Use the dedicated wrapper runner; single-run previews stop here."
    elif execution_status == "ready_for_replication_runner":
        wizard_status = "replication_runner_ready"
        continue_in_single_run = False
        message = "Route is replication-owned and has a runner contract. Use execute_replication; single-run previews stop here."
    elif route_owner == "wrapper":
        wizard_status = "wrapper_not_supported"
        continue_in_single_run = False
        message = "Route is wrapper-owned but has no executable wrapper runner contract in the current runtime."
    else:
        wizard_status = "blocked_or_nonexecutable"
        continue_in_single_run = False
        message = _planned_branch_message(warnings) or "; ".join(blocked_reasons or warnings) or "Route is not executable in the current single-run surface."

    return {
        "route_owner": route_owner,
        "execution_status": execution_status,
        "wizard_status": wizard_status,
        "continue_in_single_run": continue_in_single_run,
        "message": message,
        "warnings": warnings,
        "blocked_reasons": blocked_reasons,
        "tree_context_summary": _tree_context_summary(tree_context) if tree_context else "",
        "wrapper_handoff": dict(compile_manifest.get("wrapper_handoff", {})),
    }


def _draft_route_preview(recipe: dict[str, Any], error: str) -> dict[str, Any]:
    research_design = _recipe_fixed(recipe, "0_meta").get("research_design", "single_forecast_run")
    target_structure = _target_structure_value(recipe)
    explicit_unit = _recipe_fixed(recipe, "0_meta").get("experiment_unit")
    route_owner = "wrapper" if research_design == "study_bundle" else "single_run"
    if explicit_unit in {"single_target_full_sweep", "multi_target_separate_runs", "multi_target_shared_design", "benchmark_suite", "ablation_study"}:
        route_owner = "wrapper"
    elif explicit_unit == "replication_recipe":
        route_owner = "replication"
    if route_owner == "wrapper":
        wizard_status = "wrapper_not_supported"
        continue_in_single_run = False
        message = "Wrapper-owned route chosen, but no executable wrapper runner contract is available for this draft."
    else:
        wizard_status = "draft_incomplete"
        continue_in_single_run = True
        message = error
    return {
        "route_owner": route_owner,
        "execution_status": "draft_incomplete",
        "wizard_status": wizard_status,
        "continue_in_single_run": continue_in_single_run,
        "message": message,
        "warnings": [error],
        "blocked_reasons": [],
        "tree_context_summary": f"route_owner={route_owner}; research_design={research_design}; target_structure={target_structure}",
        "wrapper_handoff": {},
    }


def _preview_recipe_dict(recipe: dict[str, Any]) -> dict[str, Any]:
    try:
        compile_result = compile_recipe_dict(recipe)
    except CompileValidationError as exc:
        return {
            "compile_preview": None,
            "tree_context": {},
            "route_preview": _draft_route_preview(recipe, str(exc)),
        }
    compile_manifest = compile_result.manifest
    return {
        "compile_preview": compile_manifest,
        "tree_context": dict(compile_manifest.get("tree_context", {})),
        "route_preview": _route_preview(compile_manifest),
    }


def _runs_preview(compile_manifest: dict[str, Any], *, output_root: str | Path) -> dict[str, Any]:
    run_spec = dict(compile_manifest["run_spec"])
    artifact_dir = Path(output_root) / run_spec["artifact_subdir"]
    return {
        "output_root": str(Path(output_root)),
        "artifact_subdir": run_spec["artifact_subdir"],
        "artifact_dir_preview": str(artifact_dir),
        "run_id": run_spec["run_id"],
        "route_owner": run_spec["route_owner"],
    }


def _manifest_preview(compile_manifest: dict[str, Any], *, output_root: str | Path) -> dict[str, Any]:
    tree_context = dict(compile_manifest.get("tree_context", {}))
    leaf_config = dict(tree_context.get("leaf_config", compile_manifest.get("leaf_config", {})))
    stat_test_spec = dict(compile_manifest.get("stat_test_spec", {}))
    active_stat_tests = active_stat_test_axes(stat_test_spec)
    importance_spec = dict(compile_manifest.get("importance_spec", {}))
    importance_methods = active_importance_methods(importance_spec)
    expected_artifacts = [
        "manifest.json",
        "summary.txt",
        "data_preview.csv",
        "predictions.csv",
        "metrics.json",
        "comparison_summary.json",
    ]
    for stat_test in active_stat_tests.values():
        expected_artifacts.append(f"stat_test_{stat_test}.json")
    if importance_methods:
        expected_artifacts.append("importance_artifacts.json")
        expected_artifacts.extend(IMPORTANCE_FILE_NAMES.get(method, f"importance_{method}.json") for method in importance_methods)
    return {
        "recipe_id": compile_manifest["recipe_id"],
        "run_id": compile_manifest["run_spec"]["run_id"],
        "artifact_dir_preview": str(Path(output_root) / compile_manifest["run_spec"]["artifact_subdir"]),
        "route_owner": compile_manifest["run_spec"]["route_owner"],
        "target": leaf_config.get("target", ""),
        "targets": list(leaf_config.get("targets", [])),
        "horizons": list(leaf_config.get("horizons", [])),
        "benchmark_spec": dict(compile_manifest.get("benchmark_spec", {})),
        "model_spec": dict(compile_manifest.get("model_spec", {})),
        "preprocess_contract": dict(compile_manifest.get("preprocess_contract", {})),
        "tree_context": tree_context,
        "expected_artifacts": expected_artifacts,
    }


def _write_recipe_yaml(recipe: dict[str, Any], yaml_path: str | Path) -> Path:
    path = Path(yaml_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")
    return path


def _interactive_wizard(*, recipe_path: str, yaml_path: str | None, max_steps: int | None) -> dict[str, Any]:
    recipe = load_recipe_yaml(recipe_path)
    if yaml_path is None:
        yaml_path = input("YAML file path to write [custom_recipe.yaml]: ").strip() or "custom_recipe.yaml"
    recipe["recipe_id"] = Path(yaml_path).stem
    write_path = _write_recipe_yaml(recipe, yaml_path)

    completed: list[dict[str, Any]] = []
    stop_reason: str | None = None
    current_choice: dict[str, Any] | None = None
    preview = _preview_recipe_dict(recipe)
    step_count = 0

    while True:
        stack = _wizard_choice_stack(recipe)
        remaining = [choice for choice in stack if choice["key"] not in {item["key"] for item in completed}]
        if not remaining:
            current_choice = None
            break
        current_choice = remaining[0]
        if max_steps is not None and step_count >= max_steps:
            break

        current = _read_wizard_value(recipe, current_choice["key"])
        options = list(current_choice["options"])
        if options:
            option_details = current_choice.get("option_details", {})
            for idx, option in enumerate(options, 1):
                status = option_details.get(option, {}).get("status")
                suffix = f" [{status}]" if status and status != "operational" else ""
                print(f"{idx}. {option}{suffix}")
        prompt = f"{current_choice['prompt']}"
        if current is not None:
            prompt += f" [current={current}]"
        prompt += ": "
        answer = input(prompt).strip()
        if answer.lower() == "q":
            stop_reason = "Wizard stopped by user."
            break
        if answer == "":
            selected = current
        elif options and answer.isdigit() and 1 <= int(answer) <= len(options):
            selected = options[int(answer) - 1]
        elif options and answer in options:
            selected = answer
        elif options:
            selected = current
        else:
            selected = answer
        if selected is None:
            continue
        _apply_wizard_value(recipe, current_choice["key"], selected)
        write_path = _write_recipe_yaml(recipe, write_path)
        step_count += 1
        completed.append({"key": current_choice["key"], "value": _read_wizard_value(recipe, current_choice["key"])})
        preview = _preview_recipe_dict(recipe)
        if not preview["route_preview"]["continue_in_single_run"]:
            stop_reason = preview["route_preview"]["message"]
            current_choice = None
            break

    out = {
        "selected_stages": ["wizard"],
        "interactive": True,
        "yaml_path": str(write_path),
        "recipe_dict": recipe,
        "recipe_yaml": yaml.safe_dump(recipe, sort_keys=False),
        "completed_choices": completed,
        "current_choice": current_choice,
        "route_preview": preview["route_preview"],
        "stop_reason": stop_reason,
    }
    if preview["compile_preview"] is not None:
        out["compile_preview"] = preview["compile_preview"]
        out["tree_context"] = preview["tree_context"]
    return out


def macrocast_single_run(
    *,
    yaml_path: str | None = None,
    stages: str | Iterable[str] | None = None,
    output_root: str = "/tmp/macrocast_single_run_preview",
    recipe_path: str = "examples/recipes/model-benchmark.yaml",
    max_steps: int | None = None,
) -> dict[str, Any]:
    if yaml_path is None:
        return _interactive_wizard(recipe_path=recipe_path, yaml_path=yaml_path, max_steps=max_steps)

    selected = _normalize_stages(stages)
    compile_result = compile_recipe_yaml(yaml_path)
    compile_manifest = compile_result.manifest
    route_preview = _route_preview(compile_manifest)

    out: dict[str, Any] = {
        "selected_stages": selected,
        "input_yaml_path": str(Path(yaml_path)),
        "route_preview": route_preview,
    }

    if "compile_preview" in selected:
        out["compile_preview"] = compile_manifest
    if "tree_context" in selected:
        out["tree_context"] = dict(compile_manifest.get("tree_context", {}))

    blocked_preview_stages: list[str] = []
    if compile_result.compiled.execution_status != "executable":
        blocked_preview_stages = [stage for stage in selected if stage in {"runs_preview", "manifest_preview"}]
        if blocked_preview_stages:
            out["blocked_preview_stages"] = blocked_preview_stages
            out["blocked_preview_reason"] = route_preview["message"]
        return out

    if "runs_preview" in selected:
        out["runs_preview"] = _runs_preview(compile_manifest, output_root=output_root)
    if "manifest_preview" in selected:
        out["manifest_preview"] = _manifest_preview(compile_manifest, output_root=output_root)
    return out
