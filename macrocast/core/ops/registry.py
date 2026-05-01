from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from ..dag import DAG, LayerId, Node, NodeRef
from ..types import DataType

RuleSeverity = Literal["hard", "soft"]
OpStatus = Literal["operational", "future", "registry_only"]
LayerScope = Literal["universal"] | tuple[LayerId, ...]
TypeSpec = type[DataType] | tuple[type[DataType], ...]


@dataclass(frozen=True)
class Rule:
    severity: RuleSeverity
    condition: Callable[[DAG, NodeRef], bool]
    message: str
    suggestion: str | None = None


@dataclass(frozen=True)
class OpSpec:
    name: str
    layer_scope: LayerScope
    input_types: dict[str, TypeSpec]
    output_type: TypeSpec
    params_schema: dict[str, dict[str, Any]] = field(default_factory=dict)
    hard_rules: tuple[Rule, ...] = ()
    soft_rules: tuple[Rule, ...] = ()
    default_figure_type: str | None = None
    status: OpStatus = "operational"
    function: Callable[..., Any] | None = None

    def available_in(self, layer_id: LayerId) -> bool:
        return self.layer_scope == "universal" or layer_id in self.layer_scope


_OPS: dict[str, OpSpec] = {}


def clear_op_registry() -> None:
    _OPS.clear()


def register_op(
    *,
    name: str,
    layer_scope: LayerScope,
    input_types: dict[str, TypeSpec],
    output_type: TypeSpec,
    params_schema: dict[str, dict[str, Any]] | None = None,
    hard_rules: tuple[Rule, ...] = (),
    soft_rules: tuple[Rule, ...] = (),
    default_figure_type: str | None = None,
    status: OpStatus = "operational",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if name in _OPS:
            raise ValueError(f"duplicate op registration for {name!r}")
        _OPS[name] = OpSpec(
            name=name,
            layer_scope=layer_scope,
            input_types=input_types,
            output_type=output_type,
            params_schema=params_schema or {},
            hard_rules=hard_rules,
            soft_rules=soft_rules,
            default_figure_type=default_figure_type,
            status=status,
            function=func,
        )
        return func

    return decorator


def get_op(name: str) -> OpSpec:
    try:
        return _OPS[name]
    except KeyError as exc:
        raise KeyError(f"unknown op {name!r}") from exc


def list_ops() -> dict[str, OpSpec]:
    return dict(_OPS)


def type_matches(actual: TypeSpec, expected: TypeSpec) -> bool:
    actual_options = actual if isinstance(actual, tuple) else (actual,)
    expected_options = expected if isinstance(expected, tuple) else (expected,)
    return any(issubclass(actual_type, expected_type) for actual_type in actual_options for expected_type in expected_options)


def node_ref(node: Node) -> NodeRef:
    return NodeRef(node.id)
