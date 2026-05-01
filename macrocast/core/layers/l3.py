from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..dag import DAG, Node, NodeRef, SourceSelector
from ..ops.registry import TypeSpec
from ..sweep import expand_sweeps as _expand_core_sweeps
from ..types import ColumnLineage, L3MetadataArtifact, PipelineDefinition, StepRef


class L3FeatureEngineering:
    """Layer 3 Feature Engineering implementation marker."""

    @classmethod
    def list_sublayers(cls) -> tuple[str, ...]:
        return ("L3.A", "L3.B", "L3.C", "L3.D")


ALLOWED_SOURCE_SUBSET_KEYS = frozenset(
    {
        "role",
        "raw",
        "variable_group",
        "variable_list",
        "pipeline_id",
    }
)
FORECAST_COMBINATION_OPS = frozenset(
    {
        "weighted_average_forecast",
        "dmsfe",
        "bma",
        "mallows_cp",
        "median_combine_forecast",
        "trimmed_mean_combine_forecast",
    }
)


@dataclass(frozen=True)
class L3Recipe:
    root: dict[str, Any]


def parse_layer_yaml(yaml_text: str, layer_id: Literal["l3"] = "l3") -> dict[str, Any]:
    if layer_id != "l3":
        raise ValueError("L3 parser only accepts layer_id='l3'")
    from ..yaml import parse_recipe_yaml

    root = parse_recipe_yaml(yaml_text)
    raw = root.get("3_feature_engineering", root)
    if not isinstance(raw, dict):
        raise ValueError("3_feature_engineering: layer YAML must be a mapping")
    return raw


def parse_recipe_yaml(yaml_text: str) -> dict[str, Any]:
    from ..yaml import parse_recipe_yaml as parse

    return parse(yaml_text)


def parse_dag_form(nodes: list[dict[str, Any]] | dict[str, Any]) -> DAG:
    layer_yaml = nodes if isinstance(nodes, dict) else {"nodes": nodes, "sinks": {}}
    if "sinks" not in layer_yaml:
        layer_yaml = {**layer_yaml, "sinks": {}}
    return normalize_to_dag_form(layer_yaml)


def normalize_to_dag_form(layer: dict[str, Any], layer_id: Literal["l3"] = "l3") -> DAG:
    if layer_id != "l3":
        raise ValueError("L3 normalizer only accepts layer_id='l3'")
    if "nodes" not in layer:
        raise ValueError("L3 supports DAG form only; fixed_axes sugar is not supported")

    raw_nodes = list(layer.get("nodes", ()))
    pipeline_endpoints = _pipeline_endpoints(raw_nodes)
    nodes: dict[str, Node] = {}
    for raw_node in raw_nodes:
        node, extra_sources = _parse_l3_node(raw_node, pipeline_endpoints)
        for source in extra_sources:
            if source.id in nodes:
                raise ValueError(f"l3.{source.id}: duplicate node id")
            nodes[source.id] = source
        if node.id in nodes:
            raise ValueError(f"l3.{node.id}: duplicate node id")
        nodes[node.id] = node

    sinks = dict(layer.get("sinks", {}) or {})
    sink_map: dict[str, str] = {}
    features = sinks.get("l3_features_v1")
    if isinstance(features, dict):
        bundle_id = "sink:l3_features_v1"
        nodes[bundle_id] = Node(
            id=bundle_id,
            type="combine",
            layer_id="l3",
            op="l3_feature_bundle",
            inputs=(
                NodeRef(features.get("X_final", "")),
                NodeRef(features.get("y_final", ""), output_port="target"),
            ),
        )
        sink_map["l3_features_v1"] = bundle_id
    elif isinstance(features, str):
        sink_map["l3_features_v1"] = features

    metadata = sinks.get("l3_metadata_v1")
    if metadata == "auto":
        metadata_id = "sink:l3_metadata_v1"
        endpoints = tuple(NodeRef(node.id) for node in nodes.values() if node.pipeline_id)
        if not endpoints:
            endpoints = tuple(
                NodeRef(node.id)
                for node in nodes.values()
                if node.type in {"step", "combine"} and not node.id.startswith("sink:")
            )
        nodes[metadata_id] = Node(
            id=metadata_id,
            type="combine",
            layer_id="l3",
            op="l3_metadata_build",
            inputs=endpoints[:1] if len(endpoints) == 1 else endpoints,
        )
        sink_map["l3_metadata_v1"] = metadata_id
    elif isinstance(metadata, str):
        sink_map["l3_metadata_v1"] = metadata

    return DAG(
        layer_id="l3",
        nodes=nodes,
        sinks=sink_map,
        layer_globals={"leaf_config": layer.get("leaf_config", {}) or {}},
    )


def validate_layer(layer: dict[str, Any] | str, recipe_context: dict[str, Any] | None = None) -> ValidationReport:
    from ..validator import Issue, Severity, ValidationReport, validate_dag

    raw = parse_layer_yaml(layer) if isinstance(layer, str) else layer
    issues: list[Issue] = []
    if "nodes" not in raw:
        return ValidationReport((_issue("l3", "L3 supports DAG form only; fixed_axes sugar is not supported"),))
    try:
        dag = normalize_to_dag_form(raw)
    except Exception as exc:
        return ValidationReport((_issue("l3", str(exc)),))

    issues.extend(_validate_source_subsets(dag))
    issues.extend(_validate_required_sinks(raw, dag))
    issues.extend(_validate_target_construction(raw, dag))
    issues.extend(_validate_cascade(raw, dag))
    issues.extend(_validate_forecast_combination_absent(dag))

    dag_result = validate_dag(dag)
    issues.extend(
        Issue("l3_dag", Severity.HARD if issue.severity == "hard" else Severity.SOFT, "dag", issue.location, issue.message)
        for issue in dag_result.issues
    )
    issues.extend(_validate_output_contract(dag, dag_result.node_output_types))
    issues.extend(_validate_horizon_set(raw, recipe_context))
    issues.extend(_validate_regime_reference(dag, recipe_context))
    issues.extend(_soft_ordering_warnings(dag))
    return ValidationReport(tuple(issues))


def validate_recipe(recipe_yaml: dict[str, Any] | str) -> ValidationReport:
    from ..validator import ValidationReport

    root = parse_recipe_yaml(recipe_yaml) if isinstance(recipe_yaml, str) else recipe_yaml
    if "3_feature_engineering" not in root:
        return ValidationReport()
    return validate_layer(root["3_feature_engineering"], recipe_context=_recipe_context(root))


def expand_sweeps(layer: dict[str, Any] | str):
    raw = parse_layer_yaml(layer) if isinstance(layer, str) else layer
    return _expand_core_sweeps({"l3": normalize_to_dag_form(raw)})


def build_metadata_artifact(recipe_or_yaml: dict[str, Any] | str) -> L3MetadataArtifact:
    raw = parse_layer_yaml(recipe_or_yaml) if isinstance(recipe_or_yaml, str) else (
        recipe_or_yaml.get("3_feature_engineering", recipe_or_yaml)
    )
    dag = normalize_to_dag_form(raw)
    lineage: dict[str, ColumnLineage] = {}
    pipelines: dict[str, PipelineDefinition] = {}
    for node in dag.nodes.values():
        if not node.pipeline_id:
            continue
        pipelines[node.pipeline_id] = PipelineDefinition(node.pipeline_id, node.id)
        lineage[f"{node.pipeline_id}_feature"] = ColumnLineage(
            column_name=f"{node.pipeline_id}_feature",
            step_chain=(StepRef(node.id, node.op, dict(node.params)),),
            pipeline_id=node.pipeline_id,
            output_type="Panel",
        )
    if not lineage:
        for node in dag.nodes.values():
            if node.type in {"step", "combine"} and node.op != "target_construction":
                lineage[f"{node.id}_feature"] = ColumnLineage(
                    column_name=f"{node.id}_feature",
                    step_chain=(StepRef(node.id, node.op, dict(node.params)),),
                    pipeline_id=node.pipeline_id or node.id,
                    output_type="Panel",
                )
                break
    return L3MetadataArtifact(column_lineage=lineage, pipeline_definitions=pipelines)


def build_cascade_chain(depth: int) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = [
        {"id": "src_x", "type": "source", "selector": {"layer_ref": "l2", "sink_name": "l2_clean_panel_v1", "subset": {"role": "predictors"}}},
        {"id": "src_y", "type": "source", "selector": {"layer_ref": "l2", "sink_name": "l2_clean_panel_v1", "subset": {"role": "target"}}},
        {"id": "p0", "type": "step", "op": "lag", "params": {"n_lag": 1}, "pipeline_id": "p0", "inputs": ["src_x"]},
    ]
    for idx in range(1, depth + 1):
        nodes.append(
            {
                "id": f"p{idx}",
                "type": "step",
                "op": "lag",
                "params": {"n_lag": 1},
                "pipeline_id": f"p{idx}",
                "inputs": [
                    {
                        "id": f"p{idx}_ref",
                        "type": "source",
                        "selector": {"layer_ref": "l3", "sink_name": "pipeline_output", "subset": {"pipeline_id": f"p{idx - 1}"}},
                    }
                ],
            }
        )
    nodes.append({"id": "y_h", "type": "step", "op": "target_construction", "params": {"mode": "point_forecast", "method": "direct", "horizon": 1}, "inputs": ["src_y"]})
    return nodes


def example_path(name: str) -> Path:
    return Path(__file__).resolve().parents[3] / "examples" / "recipes" / name


def _parse_l3_node(raw_node: dict[str, Any], pipeline_endpoints: dict[str, list[str]]) -> tuple[Node, list[Node]]:
    extra_sources: list[Node] = []
    inputs: list[NodeRef] = []
    for index, raw_ref in enumerate(raw_node.get("inputs", ())):
        if isinstance(raw_ref, str):
            inputs.append(NodeRef(raw_ref))
            continue
        if raw_ref.get("type") == "source":
            selector = _selector(raw_ref["selector"])
            if selector.layer_ref == "l3" and selector.sink_name == "pipeline_output":
                pipeline_id = selector.subset.get("pipeline_id")
                endpoints = pipeline_endpoints.get(pipeline_id, [])
                inputs.append(NodeRef(endpoints[0] if endpoints else f"__missing_pipeline__{pipeline_id}"))
                continue
            source_id = raw_ref.get("id", f"{raw_node['id']}_src_{index}")
            extra_sources.append(Node(id=source_id, type="source", layer_id="l3", op="source", selector=selector))
            inputs.append(NodeRef(source_id))
            continue
        inputs.append(NodeRef(raw_ref["node_id"], raw_ref.get("output_port", "default")))
    selector = raw_node.get("selector")
    node_type = raw_node["type"]
    return (
        Node(
            id=raw_node["id"],
            type=node_type,
            layer_id="l3",
            op=raw_node.get("op", node_type),
            params=raw_node.get("params", {}) or {},
            inputs=tuple(inputs),
            selector=_selector(selector) if selector else None,
            pipeline_id=raw_node.get("pipeline_id"),
            status=raw_node.get("status", "operational"),
        ),
        extra_sources,
    )


def _selector(raw: dict[str, Any]) -> SourceSelector:
    return SourceSelector(raw["layer_ref"], raw["sink_name"], raw.get("subset", {}) or {})


def _pipeline_endpoints(raw_nodes: list[dict[str, Any]]) -> dict[str, list[str]]:
    endpoints: dict[str, list[str]] = {}
    for raw in raw_nodes:
        pipeline_id = raw.get("pipeline_id")
        if pipeline_id:
            endpoints.setdefault(pipeline_id, []).append(raw["id"])
    return endpoints


def _validate_source_subsets(dag: DAG) -> list[Issue]:
    issues = []
    for node in dag.nodes.values():
        if node.type != "source" or node.selector is None:
            continue
        unknown = set(node.selector.subset) - ALLOWED_SOURCE_SUBSET_KEYS
        if unknown:
            issues.append(_issue(f"l3.{node.id}", f"source selector has unknown subset keys {sorted(unknown)}"))
    return issues


def _validate_required_sinks(raw: dict[str, Any], dag: DAG) -> list[Issue]:
    issues = []
    if "l3_features_v1" not in dag.sinks:
        issues.append(_issue("l3.sinks", "L3 sink l3_features_v1 must be produced"))
    if "l3_metadata_v1" not in dag.sinks:
        issues.append(_issue("l3.sinks", "L3 sink l3_metadata_v1 must be produced"))
    features = (raw.get("sinks", {}) or {}).get("l3_features_v1")
    if isinstance(features, dict) and (not features.get("X_final") or not features.get("y_final")):
        issues.append(_issue("l3.sinks.l3_features_v1", "features sink requires X_final and y_final"))
    return issues


def _validate_target_construction(raw: dict[str, Any], dag: DAG) -> list[Issue]:
    target_nodes = [node for node in dag.nodes.values() if node.op == "target_construction"]
    issues = []
    if len(target_nodes) != 1:
        issues.append(_issue("l3.target_construction", "target_construction must appear exactly once"))
    features = (raw.get("sinks", {}) or {}).get("l3_features_v1")
    y_final = features.get("y_final") if isinstance(features, dict) else None
    for node in target_nodes:
        if node.id != y_final:
            issues.append(_issue(f"l3.{node.id}", "target_construction is only allowed for L3.A y_final"))
    return issues


def _validate_cascade(raw: dict[str, Any], dag: DAG) -> list[Issue]:
    issues = []
    endpoints = _pipeline_endpoints(list(raw.get("nodes", ())))
    for pipeline_id, node_ids in endpoints.items():
        if len(node_ids) > 1:
            issues.append(_issue(f"l3.pipeline_id.{pipeline_id}", "cascade source resolves ambiguously to multiple pipeline endpoints"))
    for raw_node in raw.get("nodes", ()):
        for raw_ref in raw_node.get("inputs", ()):
            if isinstance(raw_ref, dict) and raw_ref.get("type") == "source":
                selector = raw_ref.get("selector", {})
                if selector.get("layer_ref") == "l3" and selector.get("sink_name") == "pipeline_output":
                    pipeline_id = (selector.get("subset") or {}).get("pipeline_id")
                    if pipeline_id not in endpoints:
                        issues.append(_issue(f"l3.{raw_node['id']}", f"cascade pipeline_id {pipeline_id!r} does not exist"))
    max_depth = int((raw.get("leaf_config", {}) or {}).get("max_cascade_depth", 3))
    depth = _cascade_depth(raw)
    if depth > max_depth:
        issues.append(_issue("l3.cascade", f"cascade depth {depth} exceeds max_cascade_depth {max_depth}"))
    if _cascade_has_cycle(raw):
        issues.append(_issue("l3.cascade", "cascade graph contains a cycle"))
    return issues


def _validate_forecast_combination_absent(dag: DAG) -> list[Issue]:
    return [_issue(f"l3.{node.id}", f"forecast combination op {node.op} belongs in L4, not L3") for node in dag.nodes.values() if node.op in FORECAST_COMBINATION_OPS]


def _validate_output_contract(dag: DAG, output_types: dict[str, TypeSpec]) -> list[Issue]:
    from ..types import Panel, Series

    issues = []
    features_sink = dag.nodes.get(dag.sinks.get("l3_features_v1", ""))
    if features_sink:
        x_ref, y_ref = features_sink.inputs
        x_type = output_types.get(x_ref.node_id)
        y_type = output_types.get(y_ref.node_id)
        if x_type is not None and not _type_is_panel_like(x_type):
            issues.append(_issue("l3.X_final", "X_final must have at least 1 column and be Panel/Factor-like"))
        if y_type is not None and not _type_matches(y_type, Series):
            issues.append(_issue("l3.y_final", "y_final must be Series"))
        if x_ref.node_id.startswith("empty"):
            issues.append(_issue("l3.X_final", "X_final has 0 columns"))
    return issues


def _validate_horizon_set(raw: dict[str, Any], recipe_context: dict[str, Any] | None) -> list[Issue]:
    if not recipe_context or "horizons" not in recipe_context:
        return []
    horizons = set(recipe_context["horizons"])
    issues = []
    for node in raw.get("nodes", ()):
        if node.get("op") != "target_construction":
            continue
        horizon = (node.get("params") or {}).get("horizon")
        if isinstance(horizon, dict) and "sweep" in horizon:
            values = horizon["sweep"]
        else:
            values = [horizon]
        for value in values:
            if value not in horizons:
                issues.append(_issue(f"l3.{node['id']}", f"horizon {value} is not in L1 horizon_set"))
    return issues


def _validate_regime_reference(dag: DAG, recipe_context: dict[str, Any] | None) -> list[Issue]:
    if not recipe_context or recipe_context.get("regime_definition") != "none":
        return []
    for node in dag.nodes.values():
        if node.selector and node.selector.layer_ref == "l1" and node.selector.sink_name == "l1_regime_metadata_v1":
            return [_issue(f"l3.{node.id}", "l1_regime_metadata_v1 requires L1 regime_definition != none")]
    return []


def _soft_ordering_warnings(dag: DAG) -> list[Issue]:
    from ..validator import Issue, Severity

    warnings = []
    for node in dag.nodes.values():
        if node.op in {"log", "diff", "log_diff"}:
            for ref in node.inputs:
                upstream = dag.nodes.get(ref.node_id)
                if upstream and upstream.op == "lag":
                    warnings.append(Issue("l3_ordering", Severity.SOFT, "layer", f"l3.{node.id}", "ordering warning: stationary_transform should precede lag"))
    return warnings


def _cascade_depth(raw: dict[str, Any]) -> int:
    deps = _cascade_dependencies(raw)

    def depth(pid: str, seen: frozenset[str] = frozenset()) -> int:
        if pid in seen:
            return 999
        parents = deps.get(pid, set())
        if not parents:
            return 0
        return 1 + max(depth(parent, seen | {pid}) for parent in parents)

    return max((depth(pid) for pid in deps), default=0)


def _cascade_has_cycle(raw: dict[str, Any]) -> bool:
    return _cascade_depth(raw) >= 999


def _cascade_dependencies(raw: dict[str, Any]) -> dict[str, set[str]]:
    node_to_pipeline = {node["id"]: node.get("pipeline_id") for node in raw.get("nodes", ()) if node.get("pipeline_id")}
    deps: dict[str, set[str]] = {pid: set() for pid in node_to_pipeline.values() if pid}
    for node in raw.get("nodes", ()):
        pipeline_id = node.get("pipeline_id")
        if not pipeline_id:
            continue
        for raw_ref in node.get("inputs", ()):
            if isinstance(raw_ref, dict) and raw_ref.get("type") == "source":
                selector = raw_ref.get("selector", {})
                if selector.get("layer_ref") == "l3" and selector.get("sink_name") == "pipeline_output":
                    deps.setdefault(pipeline_id, set()).add((selector.get("subset") or {}).get("pipeline_id"))
    return deps


def _recipe_context(root: dict[str, Any]) -> dict[str, Any]:
    l1 = root.get("1_data", {}) or {}
    fixed = l1.get("fixed_axes", {}) or {}
    leaf = l1.get("leaf_config", {}) or {}
    horizons = leaf.get("target_horizons")
    horizon_set = fixed.get("horizon_set", "standard_md")
    if horizons is None:
        horizons = [1, 3, 6, 12] if horizon_set == "standard_md" else [1, 2, 4, 8]
    return {
        "horizons": tuple(horizons),
        "regime_definition": fixed.get("regime_definition", "none"),
    }


def _type_matches(actual: TypeSpec, expected: type) -> bool:
    actual_options = actual if isinstance(actual, tuple) else (actual,)
    return any(isinstance(actual_type, type) and issubclass(actual_type, expected) for actual_type in actual_options)


def _type_is_panel_like(actual: TypeSpec) -> bool:
    from ..types import Factor, LaggedPanel, Panel

    return _type_matches(actual, Panel) or _type_matches(actual, LaggedPanel) or _type_matches(actual, Factor)


def _issue(location: str, message: str) -> Issue:
    from ..validator import Issue, Severity

    return Issue("l3_contract", Severity.HARD, "layer", location, message)
