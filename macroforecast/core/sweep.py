from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import product
from typing import Any, Literal

from .cache import recipe_hash
from .dag import DAG, LayerId, Node, NodeRef
from .ops import get_op

SweepKind = Literal["param", "node_group", "external_axis"]
SweepMode = Literal["grid", "zip"]
NamingConvention = Literal["cell_id", "descriptive", "recipe_hash", "custom"]


@dataclass(frozen=True)
class SweepSpec:
    id: str
    kind: SweepKind


@dataclass(frozen=True)
class ParamSweep(SweepSpec):
    node_id: str
    param_path: str
    values: tuple[Any, ...]
    layer_id: LayerId | None = None
    kind: Literal["param"] = field(default="param", init=False)


@dataclass(frozen=True)
class NodeGroupSweep(SweepSpec):
    group_id: str
    members: tuple[NodeRef, ...]
    layer_id: LayerId | None = None
    kind: Literal["node_group"] = field(default="node_group", init=False)


@dataclass(frozen=True)
class ExternalAxisSweep(SweepSpec):
    layer_id: LayerId
    axis_name: str
    values: tuple[Any, ...]
    kind: Literal["external_axis"] = field(default="external_axis", init=False)


@dataclass(frozen=True)
class SweepCombination:
    mode: SweepMode = "grid"
    groups: tuple[str, ...] = ()


@dataclass(frozen=True)
class Cell:
    index: int
    cell_id: str
    sweep_values: dict[str, Any]
    concrete_dag: dict[LayerId, DAG]
    cache_hash: str


def collect_all_sweeps(dags: dict[LayerId, DAG]) -> tuple[SweepSpec, ...]:
    sweeps: list[SweepSpec] = []
    for layer_id, dag in dags.items():
        for node in dag.nodes.values():
            for path, value in _walk_sweep_params(node.params):
                sweep_id = f"{layer_id}.{node.id}.{path}"
                sweeps.append(
                    ParamSweep(
                        id=sweep_id,
                        layer_id=layer_id,
                        node_id=node.id,
                        param_path=path,
                        values=tuple(value["sweep"]),
                    )
                )
        for axis_name, value in dag.layer_globals.items():
            if isinstance(value, dict) and "sweep" in value:
                sweeps.append(
                    ExternalAxisSweep(
                        id=f"{layer_id}.{axis_name}",
                        layer_id=layer_id,
                        axis_name=axis_name,
                        values=tuple(value["sweep"]),
                    )
                )
        for raw_group in dag.layer_globals.get("_sweep_groups", ()):
            members = tuple(NodeRef(member) if isinstance(member, str) else member for member in raw_group.get("members", ()))
            sweeps.append(
                NodeGroupSweep(
                    id=raw_group.get("id", raw_group.get("group_id")),
                    group_id=raw_group.get("id", raw_group.get("group_id")),
                    layer_id=layer_id,
                    members=members,
                )
            )
    return tuple(sweeps)


def combine_sweeps(
    sweep_specs: tuple[SweepSpec, ...],
    combination: SweepCombination | None = None,
) -> list[dict[str, Any]]:
    if not sweep_specs:
        return [{}]
    combo = combination or SweepCombination()
    mode = combo.mode
    if mode == "grid":
        return [
            {spec.id: value for spec, value in zip(sweep_specs, values)}
            for values in product(*[_sweep_values(spec) for spec in sweep_specs])
        ]
    lengths = {len(_sweep_values(spec)) for spec in sweep_specs}
    if len(lengths) != 1:
        raise ValueError("zip sweep combination requires equal sweep lengths")
    return [
        {spec.id: _sweep_values(spec)[idx] for spec in sweep_specs}
        for idx in range(next(iter(lengths)))
    ]


def expand_sweeps(
    dags: dict[LayerId, DAG],
    combination: SweepCombination | None = None,
    naming: NamingConvention = "descriptive",
    naming_template: str | None = None,
) -> list[Cell]:
    sweep_specs = collect_all_sweeps(dags)
    values_by_cell = combine_sweeps(sweep_specs, combination)
    cells: list[Cell] = []
    for index, values in enumerate(values_by_cell, start=1):
        concrete = instantiate_cell(values, dags, sweep_specs)
        cache_hash = recipe_hash(concrete)
        cell = Cell(index=index, cell_id="", sweep_values=values, concrete_dag=concrete, cache_hash=cache_hash)
        cells.append(replace(cell, cell_id=generate_cell_id(cell, naming, naming_template)))
    return cells


def instantiate_cell(
    sweep_values: dict[str, Any],
    dags: dict[LayerId, DAG],
    sweep_specs: tuple[SweepSpec, ...] | None = None,
) -> dict[LayerId, DAG]:
    specs_by_id = {spec.id: spec for spec in (sweep_specs or collect_all_sweeps(dags))}
    concrete = {layer_id: _strip_sweep_markers(dag) for layer_id, dag in dags.items()}
    for sweep_id, value in sweep_values.items():
        spec = specs_by_id[sweep_id]
        if isinstance(spec, ParamSweep):
            if spec.layer_id is None:
                raise ValueError(f"{spec.id}: param sweep requires layer_id")
            concrete[spec.layer_id] = _apply_param_sweep(concrete[spec.layer_id], spec, value)
        elif isinstance(spec, ExternalAxisSweep):
            concrete[spec.layer_id] = _apply_external_axis_sweep(concrete[spec.layer_id], spec, value)
        elif isinstance(spec, NodeGroupSweep):
            concrete = _apply_node_group_sweep(concrete, spec, value)
    return concrete


def generate_cell_id(
    cell: Cell,
    naming: NamingConvention = "descriptive",
    template: str | None = None,
    custom_callable: Any | None = None,
) -> str:
    if naming == "cell_id":
        return f"cell_{cell.index:03d}"
    if naming == "recipe_hash":
        return cell.cache_hash[:8]
    if naming == "custom":
        if custom_callable is None:
            raise ValueError("custom cell naming requires custom_callable")
        return str(custom_callable(cell))
    if template:
        return template.format(**{_safe_key(k): v for k, v in cell.sweep_values.items()})
    if not cell.sweep_values:
        return f"cell_{cell.index:03d}"
    parts = [f"{_safe_key(k)}-{_safe_value(v)}" for k, v in sorted(cell.sweep_values.items())]
    return "__".join(parts)


def validate_sweepable_params(dags: dict[LayerId, DAG]) -> None:
    for layer_id, dag in dags.items():
        for node in dag.nodes.values():
            if node.type not in {"step", "combine"}:
                continue
            try:
                op = get_op(node.op)
            except KeyError:
                continue
            for path, _value in _walk_sweep_params(node.params):
                root = path.split(".", 1)[0]
                schema = op.params_schema.get(root, {})
                if not schema.get("sweepable", False):
                    raise ValueError(f"{layer_id}.{node.id}: param {path!r} is not sweepable for op {node.op!r}")


def _sweep_values(spec: SweepSpec) -> tuple[Any, ...]:
    if isinstance(spec, (ParamSweep, ExternalAxisSweep)):
        return spec.values
    if isinstance(spec, NodeGroupSweep):
        return spec.members
    raise TypeError(f"unsupported sweep spec {type(spec).__name__}")


def _walk_sweep_params(params: dict[str, Any], prefix: str = "") -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    for key, value in params.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and "sweep" in value:
            found.append((path, value))
        elif isinstance(value, dict):
            found.extend(_walk_sweep_params(value, path))
    return found


def _strip_sweep_markers(dag: DAG) -> DAG:
    nodes = {
        node_id: replace(node, params=_replace_sweep_with_first(node.params))
        for node_id, node in dag.nodes.items()
    }
    layer_globals = _replace_sweep_with_first(dag.layer_globals)
    return replace(dag, nodes=nodes, layer_globals=layer_globals)


def _replace_sweep_with_first(value: Any) -> Any:
    if isinstance(value, dict):
        if "sweep" in value:
            sweep_values = value["sweep"]
            if not sweep_values:
                raise ValueError("sweep values cannot be empty")
            return sweep_values[0]
        return {key: _replace_sweep_with_first(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_replace_sweep_with_first(child) for child in value]
    return value


def _apply_param_sweep(dag: DAG, spec: ParamSweep, value: Any) -> DAG:
    node = dag.node(spec.node_id)
    params = _set_path(node.params, spec.param_path, value)
    nodes = dict(dag.nodes)
    nodes[node.id] = replace(node, params=params)
    return replace(dag, nodes=nodes)


def _apply_external_axis_sweep(dag: DAG, spec: ExternalAxisSweep, value: Any) -> DAG:
    layer_globals = dict(dag.layer_globals)
    layer_globals[spec.axis_name] = value
    nodes = dict(dag.nodes)
    axis_node_id = f"axis_{spec.axis_name}"
    if axis_node_id in nodes:
        nodes[axis_node_id] = replace(nodes[axis_node_id], params={"value": value})
    return replace(dag, nodes=nodes, layer_globals=layer_globals)


def _apply_node_group_sweep(
    dags: dict[LayerId, DAG],
    spec: NodeGroupSweep,
    value: Any,
) -> dict[LayerId, DAG]:
    if spec.layer_id is None or not isinstance(value, NodeRef):
        return dags
    dag = dags[spec.layer_id]
    member_ids = {member.node_id for member in spec.members}
    sinks = {
        sink_name: value.node_id if sink_name == spec.group_id or target in member_ids else target
        for sink_name, target in dag.sinks.items()
    }
    updated = dict(dags)
    updated[spec.layer_id] = replace(dag, sinks=sinks)
    return updated


def _set_path(params: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    keys = path.split(".")
    updated = dict(params)
    cursor = updated
    for key in keys[:-1]:
        cursor[key] = dict(cursor.get(key, {}))
        cursor = cursor[key]
    cursor[keys[-1]] = value
    return updated


def _safe_key(value: str) -> str:
    return value.replace(".", "_").replace("/", "_")


def _safe_value(value: Any) -> str:
    return str(value).replace(" ", "_").replace("/", "_")
