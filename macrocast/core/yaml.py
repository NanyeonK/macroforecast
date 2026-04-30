from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml

from .dag import DAG, Edge, GatePredicate, LayerId, Node, NodeRef, SourceSelector
from .layers import list_layers


LAYER_YAML_KEYS: dict[LayerId, str] = {
    "l0": "0_meta",
    "l1": "1_data",
    "l1_5": "1_5_data_summary",
    "l2": "2_preprocessing",
    "l2_5": "2_5_pre_post_preprocessing",
    "l3": "3_feature_engineering",
    "l3_5": "3_5_feature_diagnostics",
    "l4": "4_forecasting_model",
    "l4_5": "4_5_generator_diagnostics",
    "l5": "5_evaluation",
    "l6": "6_statistical_tests",
    "l7": "7_interpretation",
    "l8": "8_output",
}

YAML_LAYER_IDS = {value: key for key, value in LAYER_YAML_KEYS.items()}


@dataclass(frozen=True)
class RecipeMetadata:
    name: str = ""
    description: str = ""
    author: str = ""
    created_at: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LayerYamlSpec:
    layer_id: LayerId
    raw_yaml: dict[str, Any]
    enabled: bool = True


def parse_recipe_yaml(text: str) -> dict[str, Any]:
    loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, dict):
        raise ValueError("recipe YAML root must be a mapping")
    return loaded


def parse_dag_form(layer_yaml: dict[str, Any], layer_id: LayerId) -> DAG:
    nodes = {}
    for raw_node in layer_yaml.get("nodes", ()):
        node = _parse_node(raw_node, layer_id)
        if node.id in nodes:
            raise ValueError(f"{layer_id}.{node.id}: duplicate node id")
        nodes[node.id] = node
    edges = tuple(_parse_edge(edge) for edge in layer_yaml.get("edges", ()))
    sinks = _parse_sinks(layer_yaml.get("sinks", {}), nodes, layer_id)
    layer_globals = {
        key: value
        for key, value in layer_yaml.items()
        if key not in {"nodes", "edges", "sinks", "fixed_axes", "leaf_config", "sweep_groups"}
    }
    if "sweep_groups" in layer_yaml:
        layer_globals["_sweep_groups"] = tuple(layer_yaml["sweep_groups"])
    return DAG(layer_id=layer_id, nodes=nodes, edges=edges, sinks=sinks, layer_globals=layer_globals)


def normalize_to_dag_form(layer_yaml: dict[str, Any], layer_id: LayerId) -> DAG:
    if "nodes" in layer_yaml:
        return parse_dag_form(layer_yaml, layer_id)

    nodes: dict[str, Node] = {}
    inputs: list[NodeRef] = []
    layer_globals = {}
    for axis_name, value in layer_yaml.get("fixed_axes", {}).items():
        node_id = f"axis_{axis_name}"
        nodes[node_id] = Node(
            id=node_id,
            type="axis",
            layer_id=layer_id,
            op=axis_name,
            params={"value": value},
        )
        layer_globals[axis_name] = value
        inputs.append(NodeRef(node_id))

    aggregate = Node(
        id="meta_aggregate",
        type="combine",
        layer_id=layer_id,
        op="layer_meta_aggregate",
        params={"leaf_config": layer_yaml.get("leaf_config", {})},
        inputs=tuple(inputs),
    )
    nodes[aggregate.id] = aggregate
    layer = list_layers()[layer_id]
    sinks = {_sink_name(produced): aggregate.id for produced in layer.produces}
    return DAG(layer_id=layer_id, nodes=nodes, sinks=sinks, layer_globals=layer_globals)


def recipe_layers_from_yaml(root: dict[str, Any]) -> dict[LayerId, LayerYamlSpec]:
    layers: dict[LayerId, LayerYamlSpec] = {}
    for yaml_key, layer_id in YAML_LAYER_IDS.items():
        if yaml_key not in root:
            continue
        raw = root[yaml_key] or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{yaml_key}: layer YAML must be a mapping")
        enabled = bool(raw.get("enabled", True))
        if not enabled and layer_id.endswith("_5"):
            continue
        layers[layer_id] = LayerYamlSpec(layer_id=layer_id, raw_yaml=raw, enabled=enabled)
    return layers


def _parse_node(raw_node: dict[str, Any], layer_id: LayerId) -> Node:
    selector = raw_node.get("selector")
    return Node(
        id=raw_node["id"],
        type=raw_node["type"],
        layer_id=layer_id,
        op=raw_node.get("op", raw_node["type"]),
        params=raw_node.get("params", {}),
        inputs=tuple(_parse_node_ref(ref) for ref in raw_node.get("inputs", ())),
        selector=_parse_selector(selector) if selector else None,
        is_benchmark=bool(raw_node.get("is_benchmark", False)),
        enabled=bool(raw_node.get("enabled", True)),
        status=raw_node.get("status", "operational"),
        gates=tuple(_parse_gate(gate) for gate in raw_node.get("gates", ())),
    )


def _parse_selector(raw_selector: dict[str, Any]) -> SourceSelector:
    return SourceSelector(
        layer_ref=raw_selector["layer_ref"],
        sink_name=raw_selector["sink_name"],
        subset=raw_selector.get("subset", {}),
    )


def _parse_gate(raw_gate: dict[str, Any]) -> GatePredicate:
    return GatePredicate(kind=raw_gate["kind"], target=raw_gate["target"], value=raw_gate.get("value"))


def _parse_node_ref(raw_ref: Any) -> NodeRef:
    if isinstance(raw_ref, str):
        return NodeRef(raw_ref)
    return NodeRef(node_id=raw_ref["node_id"], output_port=raw_ref.get("output_port", "default"))


def _parse_edge(raw_edge: dict[str, Any]) -> Edge:
    return Edge(
        from_node=_parse_node_ref(raw_edge["from_node"]),
        to_node=_parse_node_ref(raw_edge["to_node"]),
        to_input_port=raw_edge.get("to_input_port", "default"),
    )


def _parse_sinks(raw_sinks: dict[str, Any], nodes: dict[str, Node], layer_id: LayerId) -> dict[str, str]:
    sinks: dict[str, str] = {}
    for sink_name, target in raw_sinks.items():
        if isinstance(target, list):
            aggregate_id = f"sink_{sink_name}_aggregate"
            nodes[aggregate_id] = Node(
                id=aggregate_id,
                type="combine",
                layer_id=layer_id,
                op="layer_meta_aggregate",
                inputs=tuple(NodeRef(item) for item in target),
            )
            sinks[sink_name] = aggregate_id
        else:
            sinks[sink_name] = target
    return sinks


def _sink_name(produced: str) -> str:
    return produced.split(".", 1)[1] if "." in produced else produced
