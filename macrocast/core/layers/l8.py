from __future__ import annotations

from dataclasses import dataclass
from string import Formatter
from typing import Any, Literal

from ..dag import DAG, Node, NodeRef, SourceSelector


@dataclass(frozen=True)
class SubLayerSpec:
    axes: tuple[str, ...]


class L8Output:
    """Layer 8 Output / Provenance implementation marker."""

    sub_layers = {
        "L8_A_export_format": SubLayerSpec(("export_format", "compression")),
        "L8_B_saved_objects": SubLayerSpec(("saved_objects", "model_artifacts_format")),
        "L8_C_provenance": SubLayerSpec(("provenance_fields", "manifest_format")),
        "L8_D_artifact_granularity": SubLayerSpec(("artifact_granularity", "naming_convention")),
    }

    @classmethod
    def list_axes(cls) -> tuple[str, ...]:
        return tuple(axis for spec in cls.sub_layers.values() for axis in spec.axes)

    @classmethod
    def list_sublayers(cls) -> tuple[str, ...]:
        return tuple(cls.sub_layers)


class L8ResolvedAxes(dict):
    def __init__(self, values: dict[str, Any], active: dict[str, bool]) -> None:
        super().__init__(values)
        self._active = active

    def get_active(self, key: str) -> bool:
        return self._active.get(key, True)


L8_AXIS_NAMES = L8Output.list_axes()
DEFAULT_TEMPLATE = "{model_family}_{forecast_strategy}_h{horizon}"

DEFAULT_AXES: dict[str, Any] = {
    "export_format": "json_csv",
    "compression": "none",
    "saved_objects": None,
    "model_artifacts_format": "pickle",
    "provenance_fields": None,
    "manifest_format": "json",
    "artifact_granularity": "per_cell",
    "naming_convention": "descriptive",
}

EXPORT_FORMATS = {"json", "csv", "parquet", "json_csv", "json_parquet", "latex_tables", "markdown_report", "html_report", "all"}
COMPRESSION = {"none", "gzip", "zip"}
MODEL_ARTIFACTS_FORMATS = {"pickle", "joblib", "onnx", "pmml"}
MANIFEST_FORMATS = {"json", "yaml", "json_lines"}
GRANULARITY = {"per_cell", "per_target", "per_horizon", "per_target_horizon", "flat"}
NAMING = {"cell_id", "descriptive", "recipe_hash", "custom"}
SAVED_OBJECTS = {
    "forecasts", "forecast_intervals", "metrics", "ranking", "decomposition", "regime_metrics", "state_metrics", "model_artifacts",
    "combination_weights", "feature_metadata", "clean_panel", "raw_panel", "diagnostics_l1_5", "diagnostics_l2_5", "diagnostics_l3_5",
    "diagnostics_l4_5", "diagnostics_all", "tests", "importance", "transformation_attribution",
}
PROVENANCE_FIELDS = (
    "recipe_yaml_full", "recipe_hash", "package_version", "python_version", "r_version", "julia_version", "dependency_lockfile",
    "git_commit_sha", "git_branch_name", "data_revision_tag", "random_seed_used", "runtime_environment", "runtime_duration",
    "cell_resolved_axes",
)


def parse_layer_yaml(yaml_text: str, layer_id: Literal["l8"] = "l8") -> dict[str, Any]:
    if layer_id != "l8":
        raise ValueError("L8 parser only accepts layer_id='l8'")
    from ..yaml import parse_recipe_yaml

    root = parse_recipe_yaml(yaml_text)
    raw = root.get("8_output", root)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("8_output: layer YAML must be a mapping")
    return raw


def parse_recipe_yaml(yaml_text_or_root: str | dict[str, Any]) -> Any:
    from ..yaml import parse_recipe_yaml as parse

    root = parse(yaml_text_or_root) if isinstance(yaml_text_or_root, str) else yaml_text_or_root
    return L8Recipe(root)


@dataclass(frozen=True)
class L8Layer:
    raw_yaml: dict[str, Any]
    dag: DAG


@dataclass(frozen=True)
class L8Recipe:
    root: dict[str, Any]

    @property
    def layers(self) -> dict[str, Any]:
        if "8_output" not in self.root:
            return {}
        raw = self.root["8_output"] or {}
        return {"l8": L8Layer(raw, normalize_to_dag_form(raw, "l8", context=_recipe_context(self.root)))}


def normalize_to_dag_form(layer: dict[str, Any] | L8Layer, layer_id: Literal["l8"] = "l8", context: dict[str, Any] | None = None) -> DAG:
    raw = layer.raw_yaml if isinstance(layer, L8Layer) else layer
    resolved = resolve_axes_from_raw(raw, context=context)
    nodes: dict[str, Node] = {
        "src_l0_meta": Node("src_l0_meta", "source", "l8", "source", selector=SourceSelector("l0", "l0_meta_v1")),
        "src_l1_data": Node("src_l1_data", "source", "l8", "source", selector=SourceSelector("l1", "l1_data_definition_v1")),
        "src_l1_regime": Node("src_l1_regime", "source", "l8", "source", selector=SourceSelector("l1", "l1_regime_metadata_v1")),
        "src_l2_clean": Node("src_l2_clean", "source", "l8", "source", selector=SourceSelector("l2", "l2_clean_panel_v1")),
        "src_l3_features": Node("src_l3_features", "source", "l8", "source", selector=SourceSelector("l3", "l3_features_v1")),
        "src_l3_metadata": Node("src_l3_metadata", "source", "l8", "source", selector=SourceSelector("l3", "l3_metadata_v1")),
        "src_l4_forecasts": Node("src_l4_forecasts", "source", "l8", "source", selector=SourceSelector("l4", "l4_forecasts_v1")),
        "src_l4_models": Node("src_l4_models", "source", "l8", "source", selector=SourceSelector("l4", "l4_model_artifacts_v1")),
        "src_l4_training": Node("src_l4_training", "source", "l8", "source", selector=SourceSelector("l4", "l4_training_metadata_v1")),
        "src_l5_eval": Node("src_l5_eval", "source", "l8", "source", selector=SourceSelector("l5", "l5_evaluation_v1")),
        "src_l6_tests": Node("src_l6_tests", "source", "l8", "source", selector=SourceSelector("l6", "l6_tests_v1")),
        "src_l7_importance": Node("src_l7_importance", "source", "l8", "source", selector=SourceSelector("l7", "l7_importance_v1")),
        "src_l7_trans_attr": Node("src_l7_trans_attr", "source", "l8", "source", selector=SourceSelector("l7", "l7_transformation_attribution_v1")),
    }
    for axis in L8_AXIS_NAMES:
        if resolved.get_active(axis):
            nodes[f"axis_{axis}"] = Node(f"axis_{axis}", "axis", "l8", axis, params={"value": resolved[axis]})
    previous = "step:collect_inputs"
    nodes[previous] = Node("step:collect_inputs", "step", "l8", "l8_collect_inputs", inputs=tuple(NodeRef(node_id) for node_id in list(nodes) if node_id.startswith("src_")))
    for step in ("export_format", "saved_objects", "provenance", "artifact_granularity"):
        node_id = f"step:{step}"
        axis_inputs = tuple(NodeRef(f"axis_{axis}") for axis in _step_axes(step) if f"axis_{axis}" in nodes)
        nodes[node_id] = Node(node_id, "step", "l8", f"l8_{step}", inputs=(NodeRef(previous),) + axis_inputs)
        previous = node_id
    nodes["sink:l8_artifacts_v1"] = Node("sink:l8_artifacts_v1", "sink", "l8", "sink", inputs=(NodeRef(previous),))
    return DAG("l8", nodes, sinks={"l8_artifacts_v1": "sink:l8_artifacts_v1"}, layer_globals={"resolved_axes": resolved})


def resolve_axes(dag_or_layer: DAG | L8Layer | dict[str, Any]) -> L8ResolvedAxes:
    if isinstance(dag_or_layer, DAG):
        return dag_or_layer.layer_globals.get("resolved_axes", resolve_axes_from_raw({}))
    raw = dag_or_layer.raw_yaml if isinstance(dag_or_layer, L8Layer) else dag_or_layer
    return resolve_axes_from_raw(raw)


def resolve_axes_from_raw(raw: dict[str, Any], context: dict[str, Any] | None = None) -> L8ResolvedAxes:
    context = context or {}
    fixed = raw.get("fixed_axes", {}) or {}
    leaf = raw.get("leaf_config", {}) or {}
    values = {axis: _copy_default(default) for axis, default in DEFAULT_AXES.items()}
    values.update(fixed)
    if values["saved_objects"] is None:
        values["saved_objects"] = _default_saved_objects(context)
    else:
        values["saved_objects"] = _expand_saved_objects(values["saved_objects"])
    if values["provenance_fields"] is None:
        values["provenance_fields"] = list(PROVENANCE_FIELDS)
    if context.get("uses_r_model") and "r_version" not in values["provenance_fields"]:
        values["provenance_fields"].append("r_version")
    leaf_config = {
        "output_directory": leaf.get("output_directory", "./macrocast_output/default_recipe/timestamp/"),
        "descriptive_naming_template": leaf.get("descriptive_naming_template", DEFAULT_TEMPLATE),
        **leaf,
    }
    values["leaf_config"] = leaf_config
    active = {axis: True for axis in L8_AXIS_NAMES}
    if "model_artifacts" not in values["saved_objects"]:
        active["model_artifacts_format"] = False
    return L8ResolvedAxes(values, active)


def validate_layer(layer: dict[str, Any] | str, context: dict[str, Any] | None = None):
    from ..validator import ValidationReport

    raw = parse_layer_yaml(layer) if isinstance(layer, str) else layer
    context = context or {}
    fixed = raw.get("fixed_axes", {}) or {}
    resolved = resolve_axes_from_raw(raw, context=context)
    issues: list[Any] = []
    for axis, value in fixed.items():
        if axis not in L8_AXIS_NAMES:
            issues.append(_issue(f"l8.{axis}", f"unknown L8 axis {axis!r}"))
        if _is_sweep(value):
            issues.append(_issue(f"l8.{axis}", "L8 axes are not sweepable"))
    issues.extend(_validate_values(resolved, raw.get("leaf_config", {}) or {}, context))
    return ValidationReport(tuple(issues))


def validate_recipe(recipe: L8Recipe | dict[str, Any] | str):
    from ..validator import ValidationReport

    obj = parse_recipe_yaml(recipe) if isinstance(recipe, (str, dict)) else recipe
    if "8_output" not in obj.root:
        return ValidationReport()
    return validate_layer(obj.root["8_output"] or {}, context=_recipe_context(obj.root))


def make_recipe_with_l6_l7_active() -> str:
    return """
6_statistical_tests:
  enabled: true
7_interpretation:
  enabled: true
8_output:
  fixed_axes: {}
"""


def make_recipe_without_ensemble() -> dict[str, Any]:
    return {"4_forecasting_model": {"nodes": [{"id": "fit_ridge", "type": "step", "op": "fit_model", "params": {"family": "ridge"}}]}, "8_output": {"fixed_axes": {}}}


def make_recipe_with_glmboost() -> str:
    return """
4_forecasting_model:
  nodes:
    - {id: fit_glmboost, type: step, op: fit_model, params: {family: glmboost}}
8_output:
  fixed_axes: {}
"""


def _default_saved_objects(context: dict[str, Any]) -> list[str]:
    objects = ["forecasts", "metrics", "ranking"]
    if context.get("forecast_object") in {"quantile", "density"}:
        objects.append("forecast_intervals")
    if context.get("l5_decomposition_active"):
        objects.append("decomposition")
    if context.get("regime_metrics_active"):
        objects.append("regime_metrics")
    if context.get("has_fred_sd"):
        objects.append("state_metrics")
    if context.get("has_ensemble"):
        objects.append("combination_weights")
    for diag in context.get("active_diagnostics", ()):
        objects.append(f"diagnostics_{diag}")
    if context.get("l6_enabled"):
        objects.append("tests")
    if context.get("l7_enabled"):
        objects.append("importance")
    if context.get("l7_transformation_attribution"):
        objects.append("transformation_attribution")
    return objects


def _expand_saved_objects(objects: list[str]) -> list[str]:
    expanded: list[str] = []
    for obj in objects:
        if obj == "diagnostics_all":
            expanded.extend(["diagnostics_l1_5", "diagnostics_l2_5", "diagnostics_l3_5", "diagnostics_l4_5"])
        else:
            expanded.append(obj)
    return expanded


def _validate_values(resolved: L8ResolvedAxes, leaf: dict[str, Any], context: dict[str, Any]) -> list[Any]:
    issues = []
    if any(_is_sweep(resolved.get(axis)) for axis in L8_AXIS_NAMES):
        return issues
    if resolved["export_format"] not in EXPORT_FORMATS:
        issues.append(_issue("l8.export_format", "unknown export_format"))
    if resolved["compression"] not in COMPRESSION:
        issues.append(_issue("l8.compression", "unknown compression"))
    if resolved["manifest_format"] not in MANIFEST_FORMATS:
        issues.append(_issue("l8.manifest_format", "unknown manifest_format"))
    if resolved["artifact_granularity"] not in GRANULARITY:
        issues.append(_issue("l8.artifact_granularity", "unknown artifact_granularity"))
    if resolved["naming_convention"] not in NAMING:
        issues.append(_issue("l8.naming_convention", "unknown naming_convention"))
    unknown = [obj for obj in resolved["saved_objects"] if obj not in SAVED_OBJECTS and not obj.startswith("diagnostics_")]
    if unknown:
        issues.append(_issue("l8.saved_objects", f"unknown saved_objects: {unknown}"))
    if resolved["model_artifacts_format"] in {"onnx", "pmml"}:
        issues.append(_issue("l8.model_artifacts_format", "onnx and pmml are future model artifact formats"))
    if resolved["naming_convention"] == "custom" and "custom_naming_function" not in leaf:
        issues.append(_issue("l8.naming_convention", "custom naming requires custom_naming_function"))
    if resolved["naming_convention"] == "descriptive":
        allowed = {"model_family", "forecast_strategy", "horizon", "combine_method", "target"}
        placeholders = {field for _, field, _, _ in Formatter().parse(resolved["leaf_config"]["descriptive_naming_template"]) if field}
        if not placeholders <= allowed:
            issues.append(_issue("l8.descriptive_naming_template", "template references unknown recipe axes"))
    if "state_metrics" in resolved["saved_objects"] and not context.get("has_fred_sd", False):
        issues.append(_issue("l8.saved_objects", "state_metrics requires FRED-SD geography active"))
    if "regime_metrics" in resolved["saved_objects"] and context.get("regime_definition", "none") == "none":
        issues.append(_issue("l8.saved_objects", "regime_metrics requires active regime"))
    if "combination_weights" in resolved["saved_objects"] and not context.get("has_ensemble", False):
        issues.append(_issue("l8.saved_objects", "combination_weights requires L4 ensemble combine"))
    if resolved["export_format"] in {"latex_tables", "markdown_report", "html_report"} and not context.get("l5_active", True):
        issues.append(_issue("l8.export_format", f"{resolved['export_format']} requires L5 active"))
    return issues


def _recipe_context(root: dict[str, Any]) -> dict[str, Any]:
    l1 = root.get("1_data", {}) or {}
    l1_fixed = l1.get("fixed_axes", {}) or {}
    l4 = root.get("4_forecasting_model", {}) or {}
    l4_nodes = l4.get("nodes", ()) or ()
    l5 = root.get("5_evaluation", {}) or {}
    l5_fixed = l5.get("fixed_axes", {}) or {}
    l7 = root.get("7_interpretation", {}) or {}
    diagnostic_layers = {
        "l1_5": "1_5_data_summary",
        "l2_5": "2_5_pre_post_preprocessing",
        "l3_5": "3_5_feature_diagnostics",
        "l4_5": "4_5_generator_diagnostics",
    }
    return {
        "forecast_object": l4.get("forecast_object", "point"),
        "has_fred_sd": l1_fixed.get("dataset", "fred_md") in {"fred_sd", "fred_md+fred_sd", "fred_qd+fred_sd"},
        "regime_definition": l1_fixed.get("regime_definition", "none"),
        "has_ensemble": any(isinstance(node, dict) and node.get("type") == "combine" for node in l4_nodes),
        "uses_r_model": any(isinstance(node, dict) and (node.get("params", {}) or {}).get("family") == "glmboost" for node in l4_nodes),
        "l5_active": bool(l5.get("enabled", True)),
        "l5_decomposition_active": l5_fixed.get("decomposition_target", "none") != "none",
        "regime_metrics_active": l5_fixed.get("regime_use", "pooled") != "pooled",
        "l6_enabled": bool((root.get("6_statistical_tests", {}) or {}).get("enabled", False)),
        "l7_enabled": bool(l7.get("enabled", False)),
        "l7_transformation_attribution": "l7_transformation_attribution_v1" in (l7.get("sinks", {}) or {}),
        "active_diagnostics": tuple(layer_id for layer_id, yaml_key in diagnostic_layers.items() if (root.get(yaml_key, {}) or {}).get("enabled", False)),
    }


def _step_axes(step: str) -> tuple[str, ...]:
    return {
        "export_format": ("export_format", "compression"),
        "saved_objects": ("saved_objects", "model_artifacts_format"),
        "provenance": ("provenance_fields", "manifest_format"),
        "artifact_granularity": ("artifact_granularity", "naming_convention"),
    }[step]


def _copy_default(value: Any) -> Any:
    return list(value) if isinstance(value, list) else value


def _is_sweep(value: Any) -> bool:
    return isinstance(value, dict) and "sweep" in value


def _issue(location: str, message: str):
    from ..validator import Issue, Severity

    return Issue("l8_contract", Severity.HARD, "layer", location, message)
