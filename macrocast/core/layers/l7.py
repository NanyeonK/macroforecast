from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..dag import DAG, Node, NodeRef, SourceSelector
from ..ops.l7_ops import FUTURE_OPS, PRE_DEFINED_BLOCKS


class L7Interpretation:
    """Layer 7 Interpretation / Importance implementation marker."""

    sub_layers = ("L7.A", "L7.B")

    @classmethod
    def list_sublayers(cls) -> tuple[str, ...]:
        return cls.sub_layers

    @classmethod
    def list_axes(cls) -> tuple[str, ...]:
        return ("enabled",) + L7_OUTPUT_AXES


class L7ResolvedAxes(dict):
    def get_active(self, key: str) -> bool:
        return True


L7_OUTPUT_AXES = (
    "output_table_format",
    "figure_type",
    "top_k_features_to_show",
    "precision_digits",
    "figure_dpi",
    "figure_format",
    "latex_table_export",
    "markdown_table_export",
)

DEFAULT_AXES: dict[str, Any] = {
    "enabled": False,
    "output_table_format": "long",
    "figure_type": "auto",
    "top_k_features_to_show": 20,
    "precision_digits": 4,
    "figure_dpi": 300,
    "figure_format": "pdf",
    "latex_table_export": True,
    "markdown_table_export": False,
}

TREE_SET = {"random_forest", "xgboost", "lightgbm", "gradient_boosting", "decision_tree", "extra_trees", "catboost"}
LINEAR_SET = {"ols", "ridge", "lasso", "elastic_net", "var", "bvar_minnesota", "bvar_normal_inverse_wishart", "ar_p", "factor_augmented_ar"}
NN_SET = {"mlp", "lstm", "gru", "transformer"}
VAR_SET = {"var", "bvar_minnesota", "bvar_normal_inverse_wishart"}
BVAR_SET = {"bvar_minnesota", "bvar_normal_inverse_wishart"}


def parse_layer_yaml(yaml_text: str, layer_id: Literal["l7"] = "l7") -> dict[str, Any]:
    if layer_id != "l7":
        raise ValueError("L7 parser only accepts layer_id='l7'")
    from ..yaml import parse_recipe_yaml

    root = parse_recipe_yaml(yaml_text)
    raw = root.get("7_interpretation", root)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("7_interpretation: layer YAML must be a mapping")
    return raw


def parse_recipe_yaml(yaml_text_or_root: str | dict[str, Any]) -> Any:
    from ..yaml import parse_recipe_yaml as parse

    root = parse(yaml_text_or_root) if isinstance(yaml_text_or_root, str) else yaml_text_or_root
    return L7Recipe(root)


@dataclass(frozen=True)
class L7Layer:
    raw_yaml: dict[str, Any]
    dag: DAG


@dataclass(frozen=True)
class L7Recipe:
    root: dict[str, Any]

    @property
    def layers(self) -> dict[str, Any]:
        if "7_interpretation" not in self.root:
            return {}
        raw = self.root["7_interpretation"] or {}
        return {"l7": L7Layer(raw, normalize_to_dag_form(raw, "l7"))}


def parse_dag_form(nodes: list[dict[str, Any]] | dict[str, Any]) -> DAG:
    layer_yaml = nodes if isinstance(nodes, dict) else {"nodes": nodes, "sinks": {}}
    return normalize_to_dag_form(layer_yaml)


def normalize_to_dag_form(layer: dict[str, Any] | L7Layer, layer_id: Literal["l7"] = "l7") -> DAG:
    raw = layer.raw_yaml if isinstance(layer, L7Layer) else layer
    nodes: dict[str, Node] = {}
    for raw_node in raw.get("nodes", ()) or ():
        if not isinstance(raw_node, dict):
            continue
        node = _parse_node(raw_node)
        nodes[node.id] = node
    sink_map: dict[str, str] = {}
    for sink_name, target in (raw.get("sinks", {}) or {}).items():
        if isinstance(target, dict):
            first = next(iter(target.values()), None)
            if isinstance(first, list):
                first = first[0] if first else None
            sink_map[sink_name] = first
        else:
            sink_map[sink_name] = target
    return DAG("l7", nodes, sinks=sink_map, layer_globals={"resolved_axes": resolve_axes_from_raw(raw)})


def resolve_axes(dag_or_layer: DAG | L7Layer | dict[str, Any]) -> L7ResolvedAxes:
    if isinstance(dag_or_layer, DAG):
        return dag_or_layer.layer_globals.get("resolved_axes", L7ResolvedAxes(DEFAULT_AXES | {"leaf_config": {}}))
    raw = dag_or_layer.raw_yaml if isinstance(dag_or_layer, L7Layer) else dag_or_layer
    return resolve_axes_from_raw(raw)


def resolve_axes_from_raw(raw: dict[str, Any]) -> L7ResolvedAxes:
    values = {key: _copy_default(value) for key, value in DEFAULT_AXES.items()}
    values.update(raw.get("fixed_axes", {}) or {})
    if "enabled" in raw:
        values["enabled"] = raw["enabled"]
    values["leaf_config"] = raw.get("leaf_config", {}) or {}
    return L7ResolvedAxes(values)


def validate_layer(layer: dict[str, Any] | str, recipe_context: dict[str, Any] | None = None):
    from ..validator import Issue, Severity, ValidationReport, validate_dag

    raw = parse_layer_yaml(layer) if isinstance(layer, str) else layer
    context = recipe_context or {}
    issues: list[Issue] = []
    fixed = raw.get("fixed_axes", {}) or {}
    for axis, value in fixed.items():
        if axis not in L7_OUTPUT_AXES:
            issues.append(_issue(f"l7.{axis}", f"unknown L7 output axis {axis!r}"))
        if _is_sweep(value):
            issues.append(_issue(f"l7.{axis}", "L7.B output axes are not sweepable"))
    try:
        dag = normalize_to_dag_form(raw)
    except Exception as exc:
        return ValidationReport((_issue("l7", str(exc)),))
    issues.extend(_validate_nodes(raw, context))
    dag_result = validate_dag(dag)
    issues.extend(Issue("l7_dag", Severity.HARD if i.severity == "hard" else Severity.SOFT, "dag", i.location, i.message) for i in dag_result.issues)
    return ValidationReport(tuple(issues))


def validate_recipe(recipe: L7Recipe | dict[str, Any] | str):
    from ..validator import ValidationReport

    obj = parse_recipe_yaml(recipe) if isinstance(recipe, (str, dict)) else recipe
    if "7_interpretation" not in obj.root:
        return ValidationReport()
    return validate_layer(obj.root["7_interpretation"], recipe_context=_recipe_context(obj.root))


def execute_layer(layer: dict[str, Any] | str):
    raw = parse_layer_yaml(layer) if isinstance(layer, str) else layer
    if any(node.get("op") == "mrf_gtvp" for node in raw.get("nodes", ()) if isinstance(node, dict)):
        raise NotImplementedError("Phase 1 runtime: mrf_gtvp implementation in execution PR")
    raise NotImplementedError("Phase 1 runtime: L7 execution is deferred")


def make_l7_yaml(op: str = "shap_tree", model_family: str = "xgboost") -> str:
    return f"""
7_interpretation:
  enabled: true
  nodes:
    - {{id: src_model, type: source, selector: {{layer_ref: l4, sink_name: l4_model_artifacts_v1, subset: {{model_id: model}}}}}}
    - {{id: src_X, type: source, selector: {{layer_ref: l3, sink_name: l3_features_v1, subset: {{component: X_final}}}}}}
    - id: imp
      type: step
      op: {op}
      params: {{model_family: {model_family}}}
      inputs: [src_model, src_X]
  sinks:
    l7_importance_v1:
      global: imp
"""


def make_l7_yaml_with_lineage_attribution() -> str:
    return """
7_interpretation:
  enabled: true
  nodes:
    - {id: src_model, type: source, selector: {layer_ref: l4, sink_name: l4_model_artifacts_v1, subset: {model_id: xgb_full}}}
    - {id: src_X, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: X_final}}}
    - {id: src_l3_meta, type: source, selector: {layer_ref: l3, sink_name: l3_metadata_v1}}
    - {id: shap, type: step, op: shap_tree, params: {model_family: xgboost}, inputs: [src_model, src_X]}
    - {id: lineage, type: step, op: lineage_attribution, params: {level: pipeline_name}, inputs: [shap, src_l3_meta]}
  sinks:
    l7_importance_v1: {lineage: lineage}
"""


def _parse_node(raw_node: dict[str, Any]) -> Node:
    node_type = raw_node.get("type", "step")
    selector = None
    if node_type == "source":
        raw_selector = raw_node.get("selector", {}) or {}
        selector = SourceSelector(raw_selector.get("layer_ref"), raw_selector.get("sink_name"), raw_selector.get("subset", {}) or {})
    inputs = []
    for item in raw_node.get("inputs", ()) or ():
        if isinstance(item, str):
            inputs.append(NodeRef(item))
        elif isinstance(item, dict) and item.get("type") == "source":
            inline_id = item.get("id", f"{raw_node.get('id')}_inline_source")
            inputs.append(NodeRef(inline_id))
        elif isinstance(item, dict) and "id" in item:
            inputs.append(NodeRef(item["id"]))
    return Node(
        id=raw_node.get("id"),
        type=node_type,
        layer_id="l7",
        op=raw_node.get("op", "source" if node_type == "source" else ""),
        params=raw_node.get("params", {}) or {},
        inputs=tuple(inputs),
        selector=selector,
    )


def _validate_nodes(raw: dict[str, Any], context: dict[str, Any]) -> list[Any]:
    issues = []
    model_family = None
    for node in raw.get("nodes", ()) or ():
        if not isinstance(node, dict):
            continue
        if node.get("type") == "source" and node.get("selector", {}).get("layer_ref") == "l6":
            subset = node.get("selector", {}).get("subset", {}) or {}
            if subset.get("family") == "multiple_model" and subset.get("name") == "mcs_inclusion" and not context.get("l6_mcs_active", False):
                issues.append(_issue(f"l7.{node.get('id')}", "mcs_inclusion source requires L6.D MCS active"))
        if node.get("type") != "step":
            continue
        op = node.get("op")
        params = node.get("params", {}) or {}
        model_family = params.get("model_family", model_family or _infer_model_family(raw))
        if op in FUTURE_OPS:
            issues.append(_issue(f"l7.{node.get('id')}", f"{op} is future and cannot be used"))
        if op in {"shap_tree", "shap_interaction", "model_native_tree_importance"} and model_family not in TREE_SET:
            issues.append(_issue(f"l7.{node.get('id')}", f"{op} requires tree model"))
        if op in {"shap_linear", "model_native_linear_coef", "forecast_decomposition"} and model_family not in LINEAR_SET:
            issues.append(_issue(f"l7.{node.get('id')}", f"{op} requires linear model"))
        if op in {"shap_deep", "integrated_gradients", "saliency_map", "deep_lift", "gradient_shap"} and model_family not in NN_SET:
            issues.append(_issue(f"l7.{node.get('id')}", f"{op} requires neural-network model"))
        if op == "mrf_gtvp" and model_family != "macroeconomic_random_forest":
            issues.append(_issue(f"l7.{node.get('id')}", "mrf_gtvp requires macroeconomic_random_forest"))
        if op in {"fevd", "historical_decomposition", "generalized_irf"} and model_family not in VAR_SET:
            issues.append(_issue(f"l7.{node.get('id')}", f"{op} requires VAR/BVAR model"))
        if op == "lasso_inclusion_frequency" and model_family not in {"lasso", "elastic_net"}:
            issues.append(_issue(f"l7.{node.get('id')}", "lasso_inclusion_frequency requires lasso or elastic_net"))
        if op == "bvar_pip" and model_family not in BVAR_SET:
            issues.append(_issue(f"l7.{node.get('id')}", "bvar_pip requires BVAR model"))
        if op == "group_aggregate":
            issues.extend(_validate_grouping(node, context))
    return issues


def _validate_grouping(node: dict[str, Any], context: dict[str, Any]) -> list[Any]:
    grouping = (node.get("params", {}) or {}).get("grouping")
    dataset = context.get("dataset", "fred_md")
    issues = []
    if grouping == "mccracken_ng_md_groups" and "fred_md" not in dataset:
        issues.append(_issue(f"l7.{node.get('id')}.grouping", "mccracken_ng_md_groups requires fred_md dataset"))
    if grouping == "mccracken_ng_qd_groups" and "fred_qd" not in dataset:
        issues.append(_issue(f"l7.{node.get('id')}.grouping", "mccracken_ng_qd_groups requires fred_qd dataset"))
    if grouping == "fred_sd_states" and "fred_sd" not in dataset:
        issues.append(_issue(f"l7.{node.get('id')}.grouping", "fred_sd_states requires fred_sd dataset"))
    variables = set(context.get("variables", ()))
    required = set(PRE_DEFINED_BLOCKS.get(grouping, ()))
    if required and variables and not required <= variables:
        missing = sorted(required - variables)
        issues.append(_issue(f"l7.{node.get('id')}.grouping", f"{grouping} missing required series: {missing}"))
    return issues


def _recipe_context(root: dict[str, Any]) -> dict[str, Any]:
    l1_fixed = ((root.get("1_data", {}) or {}).get("fixed_axes", {}) or {})
    l1_leaf = ((root.get("1_data", {}) or {}).get("leaf_config", {}) or {})
    l6 = root.get("6_statistical_tests", {}) or {}
    l6d = ((l6.get("sub_layers", {}) or {}).get("L6_D_multiple_model", {}) or {})
    return {
        "dataset": l1_fixed.get("dataset", "fred_md"),
        "variables": tuple(l1_leaf.get("variable_universe_columns", ())),
        "l6_mcs_active": bool(l6.get("enabled", False)) and bool(l6d.get("enabled", False)),
    }


def _infer_model_family(raw: dict[str, Any]) -> str:
    for node in raw.get("nodes", ()) or ():
        if isinstance(node, dict) and node.get("type") == "source":
            model_id = ((node.get("selector", {}) or {}).get("subset", {}) or {}).get("model_id", "")
            if "ridge" in model_id:
                return "ridge"
            if "mrf" in model_id:
                return "macroeconomic_random_forest"
    return "xgboost"


def _copy_default(value: Any) -> Any:
    return list(value) if isinstance(value, list) else value


def _is_sweep(value: Any) -> bool:
    return isinstance(value, dict) and "sweep" in value


def _issue(location: str, message: str):
    from ..validator import Issue, Severity

    return Issue("l7_contract", Severity.HARD, "layer", location, message)
