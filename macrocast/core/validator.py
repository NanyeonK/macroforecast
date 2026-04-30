from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .dag import DAG, GatePredicate, Node, NodeRef, implicit_edges
from .ops import get_op
from .ops.registry import TypeSpec, type_matches
from .selectors import SourceContext, resolve_source_selector
from .types import DataType


class Severity(Enum):
    HARD = "hard"
    SOFT = "soft"
    INFO = "info"


@dataclass(frozen=True)
class Context:
    dag: DAG
    node: Node | None = None
    edge: Any | None = None
    full_recipe: Any | None = None
    runtime_state: dict[str, Any] | None = None


@dataclass(frozen=True)
class Rule:
    rule_id: str
    severity: Severity
    scope: str
    condition: Any
    message_template: str
    suggestion_template: str | None = None


@dataclass(frozen=True)
class Issue:
    rule_id: str
    severity: Severity
    scope: str
    location: str
    message: str
    suggestion: str | None = None


@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[Issue, ...] = ()

    @property
    def has_hard_errors(self) -> bool:
        return any(issue.severity is Severity.HARD for issue in self.issues)

    @property
    def hard_errors(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.severity is Severity.HARD]

    @property
    def soft_warnings(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.severity is Severity.SOFT]

    def extend(self, issues: list[Issue] | tuple[Issue, ...]) -> "ValidationReport":
        return ValidationReport(self.issues + tuple(issues))


UNIVERSAL_HARD_RULES: tuple[Rule, ...] = (
    Rule("no_cycle", Severity.HARD, "dag", lambda ctx: not _has_cycle(ctx.dag), "DAG must be acyclic"),
    Rule(
        "all_sinks_reachable",
        Severity.HARD,
        "dag",
        lambda ctx: all(_reachable_from_any_source(ctx.dag, sink) for sink in ctx.dag.sinks.values()),
        "All sinks must be reachable from at least one source",
    ),
    Rule(
        "required_inputs_filled",
        Severity.HARD,
        "node",
        lambda ctx: ctx.node is not None and (ctx.node.type in {"source", "axis"} or bool(ctx.node.inputs)),
        "Node {node_id} missing required input port default",
    ),
    Rule(
        "unique_node_ids",
        Severity.HARD,
        "dag",
        lambda ctx: len({node.id for node in ctx.dag.nodes.values()}) == len(ctx.dag.nodes),
        "Node IDs must be unique within a DAG",
    ),
    Rule(
        "input_references_resolve",
        Severity.HARD,
        "node",
        lambda ctx: ctx.node is not None and all(ref.node_id in ctx.dag.nodes for ref in ctx.node.inputs),
        "Input reference does not exist",
    ),
    Rule(
        "type_compatibility",
        Severity.HARD,
        "edge",
        lambda ctx: validate_dag(ctx.dag).valid,
        "Type mismatch",
    ),
    Rule(
        "cross_layer_sink_exists",
        Severity.HARD,
        "cross_layer",
        lambda ctx: validate_dag(ctx.dag).valid,
        "Source references a sink which is not produced",
    ),
    Rule(
        "gate_propagation",
        Severity.HARD,
        "dag",
        lambda ctx: _gate_propagation_consistent(ctx.dag),
        "Disabled node has active descendants",
    ),
)

CROSS_LAYER_RULES: tuple[Rule, ...] = (
    Rule(
        "cross_layer_source_selector_resolves",
        Severity.HARD,
        "cross_layer",
        lambda ctx: validate_dag(ctx.dag).valid,
        "Source references {layer_ref}.{sink_name} which is not produced",
    ),
)


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    location: str
    message: str
    suggestion: str | None = None

    def format(self) -> str:
        text = f"{self.location}: {self.message}"
        if self.suggestion:
            text = f"{text}. Suggestion: {self.suggestion}"
        return text


@dataclass(frozen=True)
class ValidationResult:
    issues: tuple[ValidationIssue, ...]
    node_output_types: dict[str, TypeSpec]
    disabled_nodes: frozenset[str]

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "hard" for issue in self.issues)


class DAGValidationError(ValueError):
    pass


def validate_recipe(recipe: Any) -> ValidationReport:
    report = ValidationReport()
    dags = recipe.to_dag_form() if hasattr(recipe, "to_dag_form") else recipe
    try:
        from .sweep import expand_sweeps, validate_sweepable_params

        validate_sweepable_params(dags)
        dag_sets = [cell.concrete_dag for cell in expand_sweeps(dags, getattr(recipe, "sweep_combination", None))]
    except ValueError as exc:
        return report.extend((Issue("sweep_validation", Severity.HARD, "dag", "recipe", str(exc)),))
    except Exception:
        dag_sets = [dags]

    for concrete_dags in dag_sets:
        for dag in concrete_dags.values():
            issues = _apply_universal_rules(dag, recipe)
            report = report.extend(issues)
            dag_result = validate_dag(dag)
            report = report.extend(_dag_result_to_issues(dag_result))
    return report


def validate_dag(dag: DAG, source_context: SourceContext | None = None) -> ValidationResult:
    issues: list[ValidationIssue] = []
    disabled_nodes = _disabled_nodes(dag)
    output_types: dict[str, TypeSpec] = {}
    resolving: set[str] = set()

    for node_id, node in dag.nodes.items():
        location = f"{node.layer_id}.{node_id}"
        if node_id in disabled_nodes:
            continue
        if node.status != "operational":
            issues.append(ValidationIssue("hard", location, f"node status {node.status!r} is not executable"))
            continue
        try:
            output_types[node_id] = _resolve_node_type(dag, node_id, output_types, source_context, resolving, disabled_nodes)
        except (KeyError, ValueError, TypeError) as exc:
            issues.append(ValidationIssue("hard", location, str(exc)))
            continue
        if node.type in {"step", "combine"}:
            op = get_op(node.op)
            for rule in op.hard_rules + op.soft_rules:
                try:
                    ok = rule.condition(dag, NodeRef(node.id))
                except Exception as exc:  # pragma: no cover - defensive rule wrapper
                    ok = False
                    message = f"{rule.message} ({type(exc).__name__}: {exc})"
                else:
                    message = rule.message
                if not ok:
                    issues.append(ValidationIssue(rule.severity, location, message, rule.suggestion))

    for sink_name, node_id in dag.sinks.items():
        if node_id not in dag.nodes:
            issues.append(ValidationIssue("hard", f"{dag.layer_id}.{sink_name}", f"sink points to unknown node {node_id!r}"))
        elif node_id in disabled_nodes:
            issues.append(ValidationIssue("hard", f"{dag.layer_id}.{sink_name}", f"sink points to disabled node {node_id!r}"))

    _check_cycles(dag, issues)
    return ValidationResult(tuple(issues), output_types, frozenset(disabled_nodes))


def assert_valid_dag(dag: DAG, source_context: SourceContext | None = None) -> ValidationResult:
    result = validate_dag(dag, source_context)
    if not result.valid:
        raise DAGValidationError("\n".join(issue.format() for issue in result.issues if issue.severity == "hard"))
    return result


def _apply_universal_rules(dag: DAG, recipe: Any | None = None) -> list[Issue]:
    issues: list[Issue] = []
    for rule in UNIVERSAL_HARD_RULES:
        contexts: list[Context]
        if rule.scope == "node":
            contexts = [Context(dag=dag, node=node, full_recipe=recipe) for node in dag.nodes.values()]
        else:
            contexts = [Context(dag=dag, full_recipe=recipe)]
        for context in contexts:
            try:
                ok = bool(rule.condition(context))
            except Exception:
                ok = False
            if not ok:
                location = dag.layer_id if context.node is None else f"{dag.layer_id}.{context.node.id}"
                message = rule.message_template.format(node_id=context.node.id if context.node else "")
                issues.append(Issue(rule.rule_id, rule.severity, rule.scope, location, message, rule.suggestion_template))
    return issues


def _dag_result_to_issues(result: ValidationResult) -> list[Issue]:
    severity_map = {"hard": Severity.HARD, "soft": Severity.SOFT, "info": Severity.INFO}
    return [
        Issue(
            rule_id="dag_validation",
            severity=severity_map.get(issue.severity, Severity.HARD),
            scope="dag",
            location=issue.location,
            message=issue.message,
            suggestion=issue.suggestion,
        )
        for issue in result.issues
    ]


def _node_output_type(
    dag: DAG,
    node: Node,
    output_types: dict[str, TypeSpec],
    source_context: SourceContext | None,
    resolving: set[str],
    disabled_nodes: set[str],
) -> TypeSpec:
    if node.type == "source":
        if node.selector is None:
            raise ValueError("source node requires selector")
        return resolve_source_selector(node.selector, source_context)
    if node.type == "axis":
        return DataType
    if node.type == "sink":
        if len(node.inputs) != 1:
            raise ValueError("sink node requires exactly one input")
        return _input_type(dag, node.inputs[0], output_types, source_context, resolving, disabled_nodes)

    op = get_op(node.op)
    if not op.available_in(node.layer_id):
        raise ValueError(f"op {node.op!r} is not registered for {node.layer_id}")
    if not node.inputs:
        raise ValueError(f"op {node.op!r} requires at least one input")
    for input_ref in node.inputs:
        actual = _input_type(dag, input_ref, output_types, source_context, resolving, disabled_nodes)
        expected = op.input_types.get(input_ref.output_port, op.input_types.get("default"))
        if expected is None:
            raise ValueError(f"op {node.op!r} has no input port {input_ref.output_port!r}")
        if not type_matches(actual, expected):
            raise TypeError(f"input {input_ref.node_id!r} type {actual} is not compatible with {expected}")
    return op.output_type


def _resolve_node_type(
    dag: DAG,
    node_id: str,
    output_types: dict[str, TypeSpec],
    source_context: SourceContext | None,
    resolving: set[str],
    disabled_nodes: set[str],
) -> TypeSpec:
    if node_id in output_types:
        return output_types[node_id]
    if node_id in disabled_nodes:
        raise ValueError(f"input node {node_id!r} is disabled")
    if node_id in resolving:
        raise ValueError(f"DAG contains a cycle at {node_id!r}")
    resolving.add(node_id)
    try:
        node = dag.node(node_id)
        output_type = _node_output_type(dag, node, output_types, source_context, resolving, disabled_nodes)
        output_types[node_id] = output_type
        return output_type
    finally:
        resolving.remove(node_id)


def _input_type(
    dag: DAG,
    input_ref: NodeRef,
    output_types: dict[str, TypeSpec],
    source_context: SourceContext | None,
    resolving: set[str],
    disabled_nodes: set[str],
) -> TypeSpec:
    if input_ref.node_id not in dag.nodes:
        raise KeyError(f"input node {input_ref.node_id!r} is unknown")
    return _resolve_node_type(dag, input_ref.node_id, output_types, source_context, resolving, disabled_nodes)


def _disabled_nodes(dag: DAG) -> set[str]:
    disabled: set[str] = set()
    changed = True
    while changed:
        changed = False
        for node in dag.nodes.values():
            if node.id in disabled:
                continue
            if not node.enabled or any(not _gate_passes(dag, gate, disabled) for gate in node.gates):
                disabled.add(node.id)
                changed = True
                continue
            if any(input_ref.node_id in disabled for input_ref in node.inputs):
                disabled.add(node.id)
                changed = True
    return disabled


def _gate_passes(dag: DAG, gate: GatePredicate, disabled_nodes: set[str]) -> bool:
    value = _gate_target_value(dag, gate.target)
    if gate.kind == "axis_equals":
        return value == gate.value
    if gate.kind == "axis_not_equals":
        return value != gate.value
    if gate.kind == "axis_in":
        return value in gate.value
    if gate.kind == "axis_not_in":
        return value not in gate.value
    if gate.kind == "node_exists":
        return gate.target in dag.nodes and gate.target not in disabled_nodes
    if gate.kind == "layer_axis_equals":
        return value == gate.value
    raise ValueError(f"unknown gate kind {gate.kind!r}")


def _gate_target_value(dag: DAG, target: str) -> Any:
    if target in dag.layer_globals:
        return dag.layer_globals[target]
    if "." in target:
        _, axis_name = target.split(".", 1)
        return dag.layer_globals.get(axis_name)
    node = dag.nodes.get(target)
    if node is not None:
        return node.params.get("value", node.op)
    return None


def _check_cycles(dag: DAG, issues: list[ValidationIssue]) -> None:
    graph: dict[str, list[str]] = {node_id: [] for node_id in dag.nodes}
    for edge in implicit_edges(dag):
        if edge.from_node.node_id in graph and edge.to_node.node_id in graph:
            graph[edge.from_node.node_id].append(edge.to_node.node_id)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            issues.append(ValidationIssue("hard", f"{dag.layer_id}.{node_id}", "DAG contains a cycle"))
            return
        if node_id in visited:
            return
        visiting.add(node_id)
        for child in graph[node_id]:
            visit(child)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in graph:
        visit(node_id)


def _has_cycle(dag: DAG) -> bool:
    issues: list[ValidationIssue] = []
    _check_cycles(dag, issues)
    return bool(issues)


def _reachable_from_any_source(dag: DAG, sink_node_id: str) -> bool:
    reverse_edges: dict[str, list[str]] = {node_id: [] for node_id in dag.nodes}
    for edge in implicit_edges(dag):
        if edge.to_node.node_id in reverse_edges:
            reverse_edges[edge.to_node.node_id].append(edge.from_node.node_id)
    seen: set[str] = set()

    def visit(node_id: str) -> bool:
        if node_id in seen:
            return False
        seen.add(node_id)
        node = dag.nodes.get(node_id)
        if node is None:
            return False
        if node.type in {"source", "axis"}:
            return True
        return any(visit(parent) for parent in reverse_edges.get(node_id, ()))

    return visit(sink_node_id)


def _gate_propagation_consistent(dag: DAG) -> bool:
    disabled = _disabled_nodes(dag)
    for node in dag.nodes.values():
        if node.id in disabled:
            continue
        if any(ref.node_id in disabled for ref in node.inputs):
            return False
    return True
