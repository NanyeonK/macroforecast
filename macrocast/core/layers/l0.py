from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from ..dag import DAG, Node, NodeRef
from ..layer_specs import AxisSpec, LayerImplementationSpec, Option, SubLayerSpec
from ..types import DataType
from ..validator import Issue, Severity, ValidationReport
from ..yaml import LayerYamlSpec, parse_recipe_yaml


FailurePolicy = Literal["fail_fast", "continue_on_failure"]
ReproducibilityMode = Literal["seeded_reproducible", "exploratory"]
ComputeMode = Literal["serial", "parallel"]
ParallelUnit = Literal["models", "horizons", "targets", "oos_dates"]

FAILURE_POLICY_OPTIONS: tuple[FailurePolicy, ...] = ("fail_fast", "continue_on_failure")
REPRODUCIBILITY_MODE_OPTIONS: tuple[ReproducibilityMode, ...] = ("seeded_reproducible", "exploratory")
COMPUTE_MODE_OPTIONS: tuple[ComputeMode, ...] = ("serial", "parallel")
PARALLEL_UNIT_OPTIONS: tuple[ParallelUnit, ...] = ("models", "horizons", "targets", "oos_dates")
L0_AXIS_NAMES: tuple[str, ...] = ("failure_policy", "reproducibility_mode", "compute_mode")

DEFAULT_FIXED_AXES: dict[str, str] = {
    "failure_policy": "fail_fast",
    "reproducibility_mode": "seeded_reproducible",
    "compute_mode": "serial",
}


@dataclass(frozen=True)
class L0MetaArtifact(DataType):
    failure_policy: FailurePolicy
    reproducibility_mode: ReproducibilityMode
    compute_mode: ComputeMode
    random_seed: int | None
    parallel_unit: ParallelUnit | None
    n_workers: int | Literal["auto"] | None
    gpu_deterministic: bool
    derived_study_scope: str
    derived_execution_route: str = "comparison_sweep"


@dataclass(frozen=True)
class ResolvedAxis:
    value: Any
    source: Literal["explicit", "derived", "dynamic_default", "package_default"]


@dataclass(frozen=True)
class L0LayerExecutionRecord:
    layer_id: Literal["l0"]
    started_at: datetime
    finished_at: datetime
    duration_seconds: float
    status: Literal["completed", "failed", "skipped_disabled", "skipped_diagnostic_off"]
    artifact: L0MetaArtifact
    resolved_axes: dict[str, ResolvedAxis]
    derived: dict[str, str]
    nodes_executed: int = 0
    nodes_cache_hit: int = 0
    nodes_cache_miss: int = 0
    produced_sinks: tuple[str, ...] = ("l0_meta_v1",)
    sink_hashes: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    traceback: str | None = None


@dataclass(frozen=True)
class L0Manifest:
    layer_execution_log: dict[str, L0LayerExecutionRecord]


@dataclass(frozen=True)
class L0Recipe:
    layer: LayerYamlSpec
    targets: tuple[str, ...] = ("target",)
    methods: tuple[str, ...] = ("method",)


def parse_layer_yaml(yaml_text: str, layer_id: Literal["l0"] = "l0") -> LayerYamlSpec:
    if layer_id != "l0":
        raise ValueError("L0 parser only accepts layer_id='l0'")
    root = parse_recipe_yaml(yaml_text)
    raw = root.get("0_meta", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("0_meta: layer YAML must be a mapping")
    return LayerYamlSpec(layer_id="l0", raw_yaml=raw, enabled=bool(raw.get("enabled", True)))


def normalize_to_dag_form(layer: LayerYamlSpec | dict[str, Any], layer_id: Literal["l0"] = "l0") -> DAG:
    if layer_id != "l0":
        raise ValueError("L0 normalizer only accepts layer_id='l0'")
    raw = _raw_layer(layer)
    fixed_axes = raw.get("fixed_axes", {}) or {}
    leaf_config = raw.get("leaf_config", {}) or {}

    nodes: dict[str, Node] = {}
    inputs: list[NodeRef] = []
    layer_globals: dict[str, Any] = {}
    for axis_name in L0_AXIS_NAMES:
        value = fixed_axes.get(axis_name, DEFAULT_FIXED_AXES[axis_name])
        node_id = f"axis_{axis_name}"
        nodes[node_id] = Node(
            id=node_id,
            type="axis",
            layer_id="l0",
            op=axis_name,
            params={"value": value},
        )
        inputs.append(NodeRef(node_id))
        layer_globals[axis_name] = value

    aggregate = Node(
        id="meta_aggregate",
        type="combine",
        layer_id="l0",
        op="layer_meta_aggregate",
        params={"leaf_config": leaf_config},
        inputs=tuple(inputs),
    )
    nodes[aggregate.id] = aggregate
    return DAG(layer_id="l0", nodes=nodes, sinks={"l0_meta_v1": "meta_aggregate"}, layer_globals=layer_globals)


def resolve_axes(dag: DAG, recipe_context: dict[str, Any] | None = None) -> dict[str, Any]:
    axis_values = _axis_values_from_dag(dag)
    leaf_config = _leaf_config_from_dag(dag)
    study_scope = derive_study_scope(recipe_context or {})

    resolved = {
        "failure_policy": axis_values["failure_policy"],
        "reproducibility_mode": axis_values["reproducibility_mode"],
        "compute_mode": axis_values["compute_mode"],
        "random_seed": leaf_config.get("random_seed"),
        "parallel_unit": leaf_config.get("parallel_unit"),
        "n_workers": leaf_config.get("n_workers"),
        "gpu_deterministic": leaf_config.get("gpu_deterministic", False),
        "derived_study_scope": study_scope,
        "derived_execution_route": "comparison_sweep",
    }
    if resolved["reproducibility_mode"] == "seeded_reproducible" and resolved["random_seed"] is None:
        resolved["random_seed"] = 42
    return resolved


def validate_layer(layer: LayerYamlSpec | dict[str, Any] | str) -> ValidationReport:
    if isinstance(layer, str):
        layer = parse_layer_yaml(layer)
    raw = _raw_layer(layer)
    issues: list[Issue] = []

    fixed_axes = raw.get("fixed_axes", {}) or {}
    leaf_config = raw.get("leaf_config", {}) or {}
    if not isinstance(fixed_axes, dict):
        return _report("l0.fixed_axes", "fixed_axes must be a mapping")
    if not isinstance(leaf_config, dict):
        return _report("l0.leaf_config", "leaf_config must be a mapping")

    for axis_name in fixed_axes:
        if axis_name not in L0_AXIS_NAMES:
            issues.append(_issue("l0.fixed_axes", f"unknown L0 axis {axis_name!r}"))

    for axis_name in L0_AXIS_NAMES:
        value = fixed_axes.get(axis_name, DEFAULT_FIXED_AXES[axis_name])
        if _is_sweep_marker(value):
            issues.append(_issue(f"l0.{axis_name}", f"L0 axis {axis_name} is not sweepable"))
            continue
        allowed = _allowed_values(axis_name)
        if value not in allowed:
            issues.append(_issue(f"l0.{axis_name}", f"{axis_name} must be one of {sorted(allowed)}"))

    failure_policy = fixed_axes.get("failure_policy", DEFAULT_FIXED_AXES["failure_policy"])
    reproducibility_mode = fixed_axes.get("reproducibility_mode", DEFAULT_FIXED_AXES["reproducibility_mode"])
    compute_mode = fixed_axes.get("compute_mode", DEFAULT_FIXED_AXES["compute_mode"])

    if not _is_sweep_marker(failure_policy) and failure_policy in FAILURE_POLICY_OPTIONS:
        pass

    if reproducibility_mode == "seeded_reproducible":
        random_seed = leaf_config.get("random_seed", 42)
        if not isinstance(random_seed, int):
            issues.append(_issue("l0.random_seed", "leaf_config.random_seed must be int for seeded_reproducible"))
    elif reproducibility_mode == "exploratory" and "random_seed" in leaf_config:
        issues.append(_issue("l0.random_seed", "leaf_config.random_seed must not be present in exploratory mode"))

    if compute_mode == "parallel":
        if "parallel_unit" not in leaf_config:
            issues.append(_issue("l0.parallel_unit", "compute_mode=parallel requires leaf_config.parallel_unit"))
        elif leaf_config["parallel_unit"] not in PARALLEL_UNIT_OPTIONS:
            issues.append(
                _issue("l0.parallel_unit", f"parallel_unit must be one of {sorted(PARALLEL_UNIT_OPTIONS)}")
            )
        if "n_workers" not in leaf_config:
            issues.append(_issue("l0.n_workers", "compute_mode=parallel requires leaf_config.n_workers"))
        elif not _valid_n_workers(leaf_config["n_workers"]):
            issues.append(_issue("l0.n_workers", "n_workers must be a positive int or 'auto'"))
    elif compute_mode == "serial":
        for key in ("parallel_unit", "n_workers"):
            if key in leaf_config:
                issues.append(_issue(f"l0.{key}", f"leaf_config.{key} must not be present when compute_mode=serial"))

    if "gpu_deterministic" in leaf_config and not isinstance(leaf_config["gpu_deterministic"], bool):
        issues.append(_issue("l0.gpu_deterministic", "leaf_config.gpu_deterministic must be bool"))

    return ValidationReport(tuple(issues))


def build_recipe_with_l0_only(yaml_text: str) -> L0Recipe:
    return L0Recipe(layer=parse_layer_yaml(yaml_text))


def build_minimal_recipe(targets: list[str], methods: list[str]) -> L0Recipe:
    return L0Recipe(layer=parse_layer_yaml("0_meta:\n  fixed_axes: {}"), targets=tuple(targets), methods=tuple(methods))


def execute_recipe(recipe: L0Recipe) -> L0Manifest:
    report = validate_layer(recipe.layer)
    if report.has_hard_errors:
        messages = "; ".join(issue.message for issue in report.hard_errors)
        raise ValueError(messages)

    started = datetime.now(UTC)
    dag = normalize_to_dag_form(recipe.layer)
    context = {"targets": recipe.targets, "methods": recipe.methods}
    resolved = resolve_axes(dag, context)
    raw = recipe.layer.raw_yaml
    fixed_axes = raw.get("fixed_axes", {}) or {}
    leaf_config = raw.get("leaf_config", {}) or {}
    resolved_axes = _resolved_axis_entries(resolved, fixed_axes, leaf_config)
    artifact = L0MetaArtifact(
        failure_policy=resolved["failure_policy"],
        reproducibility_mode=resolved["reproducibility_mode"],
        compute_mode=resolved["compute_mode"],
        random_seed=resolved["random_seed"],
        parallel_unit=resolved["parallel_unit"],
        n_workers=resolved["n_workers"],
        gpu_deterministic=resolved["gpu_deterministic"],
        derived_study_scope=resolved["derived_study_scope"],
        derived_execution_route=resolved["derived_execution_route"],
    )
    finished = datetime.now(UTC)
    record = L0LayerExecutionRecord(
        layer_id="l0",
        started_at=started,
        finished_at=finished,
        duration_seconds=(finished - started).total_seconds(),
        status="completed",
        artifact=artifact,
        resolved_axes=resolved_axes,
        derived={
            "study_scope": resolved["derived_study_scope"],
            "execution_route": resolved["derived_execution_route"],
        },
        nodes_executed=len(dag.nodes),
        nodes_cache_miss=len(dag.nodes),
    )
    return L0Manifest(layer_execution_log={"l0": record})


def derive_study_scope(recipe_context: dict[str, Any]) -> str:
    targets = recipe_context.get("targets") or ("target",)
    methods = recipe_context.get("methods") or ("method",)
    target_count = len(targets)
    method_count = len(methods)
    if target_count == 1 and method_count == 1:
        return "one_target_one_method"
    if target_count == 1:
        return "one_target_compare_methods"
    if method_count == 1:
        return "multiple_targets_one_method"
    return "multiple_targets_compare_methods"


L0_LAYER_SPEC = LayerImplementationSpec(
    layer_id="l0",
    name="Study Setup",
    category="setup",
    produces=("l0_meta_v1",),
    ui_mode="list",
    layer_globals=(),
    sub_layers=(SubLayerSpec(id="l0_a", name="Execution policy", axes=L0_AXIS_NAMES),),
    axes={
        "l0_a": {
            "failure_policy": AxisSpec(
                name="failure_policy",
                options=(
                    Option("fail_fast", "Fail fast", "Stop on first cell failure."),
                    Option("continue_on_failure", "Continue on failure", "Record failed cells and continue."),
                ),
                default="fail_fast",
                sweepable=False,
            ),
            "reproducibility_mode": AxisSpec(
                name="reproducibility_mode",
                options=(
                    Option("seeded_reproducible", "Seeded reproducible", "Use a fixed random seed."),
                    Option("exploratory", "Exploratory", "Do not fix stochastic seeds."),
                ),
                default="seeded_reproducible",
                sweepable=False,
                leaf_config_keys=("random_seed",),
            ),
            "compute_mode": AxisSpec(
                name="compute_mode",
                options=(
                    Option("serial", "Serial", "Use one worker."),
                    Option("parallel", "Parallel", "Use multiple workers over a leaf_config unit."),
                ),
                default="serial",
                sweepable=False,
                leaf_config_keys=("parallel_unit", "n_workers"),
            ),
        }
    },
)


def _raw_layer(layer: LayerYamlSpec | dict[str, Any]) -> dict[str, Any]:
    if isinstance(layer, LayerYamlSpec):
        return layer.raw_yaml
    return layer


def _axis_values_from_dag(dag: DAG) -> dict[str, Any]:
    return {axis: dag.nodes[f"axis_{axis}"].params["value"] for axis in L0_AXIS_NAMES}


def _leaf_config_from_dag(dag: DAG) -> dict[str, Any]:
    return dict(dag.nodes["meta_aggregate"].params.get("leaf_config", {}))


def _resolved_axis_entries(
    resolved: dict[str, Any], fixed_axes: dict[str, Any], leaf_config: dict[str, Any]
) -> dict[str, ResolvedAxis]:
    entries: dict[str, ResolvedAxis] = {}
    for axis_name in L0_AXIS_NAMES:
        source = "explicit" if axis_name in fixed_axes else "package_default"
        entries[axis_name] = ResolvedAxis(resolved[axis_name], source)
    for key in ("random_seed", "parallel_unit", "n_workers", "gpu_deterministic"):
        if resolved[key] is None and key not in leaf_config:
            continue
        source = "explicit" if key in leaf_config else "package_default"
        entries[key] = ResolvedAxis(resolved[key], source)
    entries["study_scope"] = ResolvedAxis(resolved["derived_study_scope"], "derived")
    entries["execution_route"] = ResolvedAxis(resolved["derived_execution_route"], "derived")
    return entries


def _allowed_values(axis_name: str) -> set[str]:
    if axis_name == "failure_policy":
        return set(FAILURE_POLICY_OPTIONS)
    if axis_name == "reproducibility_mode":
        return set(REPRODUCIBILITY_MODE_OPTIONS)
    if axis_name == "compute_mode":
        return set(COMPUTE_MODE_OPTIONS)
    raise KeyError(axis_name)


def _is_sweep_marker(value: Any) -> bool:
    return isinstance(value, dict) and "sweep" in value


def _valid_n_workers(value: Any) -> bool:
    return value == "auto" or (isinstance(value, int) and value > 0)


def _issue(location: str, message: str) -> Issue:
    return Issue("l0_contract", Severity.HARD, "layer", location, message)


def _report(location: str, message: str) -> ValidationReport:
    return ValidationReport((_issue(location, message),))
