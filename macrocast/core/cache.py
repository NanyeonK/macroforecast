from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import pickle
from typing import Any

from .dag import DAG, LayerId, Node
from .ops import get_op


def canonical_dict(value: Any) -> Any:
    if is_dataclass(value):
        return canonical_dict(asdict(value))
    if isinstance(value, dict):
        return {str(key): canonical_dict(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [canonical_dict(item) for item in value]
    if isinstance(value, float):
        return float(f"{value:.12g}")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    return value


def canonical_serialize(value: Any) -> str:
    return json.dumps(canonical_dict(value), sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any, length: int = 16) -> str:
    return sha256(canonical_serialize(value).encode("utf-8")).hexdigest()[:length]


def node_hash(
    node: Node,
    dag: DAG,
    runtime_context: dict[str, Any] | None = None,
    memo: dict[str, str] | None = None,
) -> str:
    runtime = runtime_context or {}
    cache = memo if memo is not None else {}
    if node.id in cache:
        return cache[node.id]
    input_hashes = [node_hash(dag.node(ref.node_id), dag, runtime, cache) for ref in node.inputs]
    components = {
        "layer_id": node.layer_id,
        "node_id": node.id,
        "node_type": node.type,
        "op": node.op,
        "params": node.params,
        "selector": node.selector,
        "input_hashes": input_hashes,
        "runtime_context_keys": _filter_relevant_context(node.op, runtime),
    }
    cache[node.id] = stable_hash(components)
    return cache[node.id]


def layer_hash(layer_dag: DAG, runtime_context: dict[str, Any] | None = None) -> str:
    sink_hashes = [
        node_hash(layer_dag.node(node_id), layer_dag, runtime_context)
        for _sink_name, node_id in sorted(layer_dag.sinks.items())
    ]
    return stable_hash(sorted(sink_hashes))


def recipe_hash(
    dags: dict[LayerId, DAG],
    runtime_contexts: dict[LayerId, dict[str, Any]] | None = None,
) -> str:
    contexts = runtime_contexts or {}
    layer_hashes = [
        (layer_id, layer_hash(dag, contexts.get(layer_id, dag.layer_globals)))
        for layer_id, dag in sorted(dags.items())
    ]
    return stable_hash(layer_hashes)


def ensure_cache_layout(cache_root: Path) -> None:
    for child in ("nodes", "cells", "runtime_context"):
        (cache_root / child).mkdir(parents=True, exist_ok=True)


def execute_node(
    node: Node,
    dag: DAG,
    runtime_context: dict[str, Any],
    cache_dir: Path,
) -> Any:
    ensure_cache_layout(cache_dir)
    current_hash = node_hash(node, dag, runtime_context)
    cache_path = cache_dir / "nodes" / current_hash
    result_path = cache_path / "result.pickle"
    if result_path.exists():
        with result_path.open("rb") as fh:
            return pickle.load(fh)

    inputs = [execute_node(dag.node(ref.node_id), dag, runtime_context, cache_dir) for ref in node.inputs]
    op = get_op(node.op)
    if op.function is None:
        raise ValueError(f"{node.layer_id}.{node.id}: op {node.op!r} has no executable function")
    result = op.function(inputs, node.params)

    cache_path.mkdir(parents=True, exist_ok=True)
    with result_path.open("wb") as fh:
        pickle.dump(result, fh)
    metadata = {
        "op": node.op,
        "params": node.params,
        "input_hashes": [node_hash(dag.node(ref.node_id), dag, runtime_context) for ref in node.inputs],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (cache_path / "metadata.json").write_text(canonical_serialize(metadata), encoding="utf-8")
    (cache_path / "created_at.txt").write_text(metadata["created_at"], encoding="utf-8")
    return result


def _filter_relevant_context(op_name: str, runtime_context: dict[str, Any]) -> dict[str, Any]:
    try:
        op = get_op(op_name)
    except KeyError:
        return runtime_context
    keys = set()
    for schema in op.params_schema.values():
        keys.update(schema.get("runtime_context_keys", ()))
    if not keys:
        return runtime_context
    return {key: runtime_context[key] for key in sorted(keys) if key in runtime_context}
