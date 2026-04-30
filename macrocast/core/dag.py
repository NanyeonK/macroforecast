from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


LayerId = Literal[
    "l0",
    "l1",
    "l1_5",
    "l2",
    "l2_5",
    "l3",
    "l3_5",
    "l4",
    "l4_5",
    "l5",
    "l6",
    "l7",
    "l8",
]

LayerCategory = Literal["setup", "construction", "diagnostic", "consumption"]
NodeType = Literal["source", "axis", "step", "combine", "sink"]
NodeStatus = Literal["operational", "future", "registry_only"]
GateKind = Literal[
    "axis_equals",
    "axis_not_equals",
    "axis_in",
    "axis_not_in",
    "node_exists",
    "layer_axis_equals",
]


@dataclass(frozen=True)
class NodeRef:
    node_id: str
    output_port: str = "default"


@dataclass(frozen=True)
class Edge:
    from_node: NodeRef
    to_node: NodeRef
    to_input_port: str = "default"


@dataclass(frozen=True)
class SourceSelector:
    layer_ref: LayerId | Literal["external"]
    sink_name: str
    subset: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GatePredicate:
    kind: GateKind
    target: str
    value: Any = None


@dataclass(frozen=True)
class Node:
    id: str
    type: NodeType
    layer_id: LayerId
    op: str
    params: dict[str, Any] = field(default_factory=dict)
    inputs: tuple[NodeRef, ...] = ()
    selector: SourceSelector | None = None
    is_benchmark: bool = False
    enabled: bool = True
    status: NodeStatus = "operational"
    gates: tuple[GatePredicate, ...] = ()


@dataclass(frozen=True)
class DAG:
    layer_id: LayerId
    nodes: dict[str, Node]
    edges: tuple[Edge, ...] = ()
    sinks: dict[str, str] = field(default_factory=dict)
    layer_globals: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        node_ids = [node.id for node in self.nodes.values()]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError(f"{self.layer_id}: Node IDs must be unique within a DAG")
        mismatched = [key for key, node in self.nodes.items() if key != node.id]
        if mismatched:
            raise ValueError(f"{self.layer_id}: node mapping keys must match node ids: {mismatched}")

    def node(self, node_id: str) -> Node:
        try:
            return self.nodes[node_id]
        except KeyError as exc:
            raise KeyError(f"{self.layer_id}.{node_id}: unknown node") from exc


def implicit_edges(dag: DAG) -> tuple[Edge, ...]:
    edges: list[Edge] = list(dag.edges)
    for node in dag.nodes.values():
        for input_ref in node.inputs:
            edges.append(Edge(from_node=input_ref, to_node=NodeRef(node.id)))
    return tuple(edges)
