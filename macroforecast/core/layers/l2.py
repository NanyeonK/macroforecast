from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ..dag import DAG, GatePredicate, Node, NodeRef, SourceSelector
from ..layer_specs import AxisSpec, LayerImplementationSpec, Option, SubLayerSpec
from ..types import L2CleanPanelArtifact, Panel


class L2Preprocessing:
    """Layer 2 Preprocessing implementation marker."""

    @classmethod
    def list_axes(cls) -> tuple[str, ...]:
        return L2_AXIS_NAMES


L2_AXIS_NAMES: tuple[str, ...] = (
    "sd_series_frequency_filter",
    "mixed_frequency_representation",
    "quarterly_to_monthly_rule",
    "monthly_to_quarterly_rule",
    "transform_policy",
    "sd_tcode_policy",
    "transform_scope",
    "outlier_policy",
    "outlier_action",
    "outlier_scope",
    "imputation_policy",
    "imputation_temporal_rule",
    "imputation_scope",
    "frame_edge_policy",
    "frame_edge_scope",
)

FRED_SD_DATASETS = frozenset({"fred_sd", "fred_md+fred_sd", "fred_qd+fred_sd"})
ALL_SWEEPABLE_AXES = frozenset(L2_AXIS_NAMES)

DEFAULT_AXES: dict[str, Any] = {
    "sd_series_frequency_filter": "both",
    "mixed_frequency_representation": "calendar_aligned_frame",
    "quarterly_to_monthly_rule": "step_backward",
    "monthly_to_quarterly_rule": "quarterly_average",
    "transform_policy": "apply_official_tcode",
    "sd_tcode_policy": "none",
    "outlier_policy": "mccracken_ng_iqr",
    "outlier_action": "flag_as_nan",
    "imputation_policy": "em_factor",
    "imputation_temporal_rule": "expanding_window_per_origin",
    "frame_edge_policy": "truncate_to_balanced",
}

PIPELINE_STEPS: tuple[str, ...] = ("freq_alignment", "transform", "outlier_handle", "imputation", "frame_edge")


class L2ResolvedAxes(dict):
    def __init__(self, values: dict[str, Any], source: dict[str, str]) -> None:
        super().__init__(values)
        self.source = source


@dataclass(frozen=True)
class L2LayerExecutionRecord:
    layer_id: Literal["l2"]
    status: Literal["completed", "failed", "skipped_disabled", "skipped_diagnostic_off"]
    artifact: L2CleanPanelArtifact
    resolved_axes: dict[str, Any]
    produced_sinks: tuple[str, ...] = ("l2_clean_panel_v1",)
    sink_hashes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class L2Manifest:
    layer_execution_log: dict[str, L2LayerExecutionRecord]


@dataclass(frozen=True)
class L2Recipe:
    layer: Any


def parse_layer_yaml(yaml_text: str, layer_id: Literal["l2"] = "l2") -> Any:
    if layer_id != "l2":
        raise ValueError("L2 parser only accepts layer_id='l2'")
    from ..yaml import LayerYamlSpec, parse_recipe_yaml

    root = parse_recipe_yaml(yaml_text)
    raw = root.get("2_preprocessing", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("2_preprocessing: layer YAML must be a mapping")
    return LayerYamlSpec(layer_id="l2", raw_yaml=raw, enabled=bool(raw.get("enabled", True)))


def parse_recipe_yaml(yaml_text: str) -> dict[str, Any]:
    from ..yaml import parse_recipe_yaml as parse

    return parse(yaml_text)


def normalize_to_dag_form(
    layer: Any | dict[str, Any], layer_id: Literal["l2"] = "l2", l1_context: dict[str, Any] | None = None
) -> DAG:
    if layer_id != "l2":
        raise ValueError("L2 normalizer only accepts layer_id='l2'")
    raw = _raw_layer(layer)
    fixed_axes = raw.get("fixed_axes", {}) or {}
    leaf_config = raw.get("leaf_config", {}) or {}
    resolved = resolve_axes_from_raw(fixed_axes, leaf_config, l1_context=l1_context)

    nodes: dict[str, Node] = {
        "src_l1_data_definition": Node(
            id="src_l1_data_definition",
            type="source",
            layer_id="l2",
            op="source",
            selector=SourceSelector(layer_ref="l1", sink_name="l1_data_definition_v1"),
        )
    }
    layer_globals: dict[str, Any] = {}
    for axis_name in L2_AXIS_NAMES:
        value = resolved.get(axis_name)
        if value is None:
            continue
        node_id = f"axis_{axis_name}"
        nodes[node_id] = Node(
            id=node_id,
            type="axis",
            layer_id="l2",
            op=axis_name,
            params={"value": value, "source": resolved.source[axis_name]},
            gates=_axis_gates(axis_name),
        )
        layer_globals[axis_name] = value

    previous = "src_l1_data_definition"
    for step_id in PIPELINE_STEPS:
        axis_inputs = tuple(NodeRef(f"axis_{axis}") for axis in _step_axes(step_id) if f"axis_{axis}" in nodes)
        nodes[f"step:{step_id}"] = Node(
            id=f"step:{step_id}",
            type="step",
            layer_id="l2",
            op=step_id,
            params={"leaf_config": leaf_config},
            inputs=(NodeRef(previous),) + axis_inputs,
            gates=_step_gates(step_id),
        )
        previous = f"step:{step_id}"

    nodes["sink:l2_clean_panel_v1"] = Node(
        id="sink:l2_clean_panel_v1",
        type="sink",
        layer_id="l2",
        op="sink",
        inputs=(NodeRef(previous),),
    )
    return DAG(
        layer_id="l2",
        nodes=nodes,
        sinks={"l2_clean_panel_v1": "sink:l2_clean_panel_v1"},
        layer_globals=layer_globals,
    )


def resolve_axes(dag: DAG) -> L2ResolvedAxes:
    values = {node.id.removeprefix("axis_"): node.params["value"] for node in dag.nodes.values() if node.id.startswith("axis_")}
    source = {
        node.id.removeprefix("axis_"): node.params.get("source", "explicit")
        for node in dag.nodes.values()
        if node.id.startswith("axis_")
    }
    return L2ResolvedAxes(values, source)


def resolve_axes_from_raw(
    fixed_axes: dict[str, Any], leaf_config: dict[str, Any], l1_context: dict[str, Any] | None = None
) -> L2ResolvedAxes:
    values: dict[str, Any] = {}
    source: dict[str, str] = {}
    for axis_name in L2_AXIS_NAMES:
        if axis_name in fixed_axes:
            values[axis_name] = fixed_axes[axis_name]
            source[axis_name] = "explicit"
        elif axis_name.endswith("_scope"):
            values[axis_name] = _derived_scope(axis_name, values)
            source[axis_name] = "derived"
        elif axis_name in DEFAULT_AXES:
            values[axis_name] = DEFAULT_AXES[axis_name]
            source[axis_name] = "package_default"
    if not _l2a_active(l1_context):
        for axis_name in (
            "sd_series_frequency_filter",
            "mixed_frequency_representation",
            "quarterly_to_monthly_rule",
            "monthly_to_quarterly_rule",
        ):
            if axis_name not in fixed_axes:
                values.pop(axis_name, None)
                source.pop(axis_name, None)
        # sd_tcode_policy gate: also FRED-SD-only.
        if "sd_tcode_policy" not in fixed_axes:
            values.pop("sd_tcode_policy", None)
            source.pop("sd_tcode_policy", None)
    return L2ResolvedAxes(values, source)


def validate_layer(layer: Any | dict[str, Any] | str, l1_context: dict[str, Any] | None = None) -> Any:
    from ..validator import Issue, Severity, ValidationReport

    if isinstance(layer, str):
        layer = parse_layer_yaml(layer)
    raw = _raw_layer(layer)
    fixed_axes = raw.get("fixed_axes", {}) or {}
    leaf_config = raw.get("leaf_config", {}) or {}
    issues: list[Issue] = []
    if not isinstance(fixed_axes, dict):
        return ValidationReport((_issue("l2.fixed_axes", "fixed_axes must be a mapping"),))
    if not isinstance(leaf_config, dict):
        return ValidationReport((_issue("l2.leaf_config", "leaf_config must be a mapping"),))
    for axis_name in fixed_axes:
        if axis_name not in L2_AXIS_NAMES:
            issues.append(_issue("l2.fixed_axes", f"unknown L2 axis {axis_name!r}"))
        elif _is_sweep_marker(fixed_axes[axis_name]) and axis_name not in ALL_SWEEPABLE_AXES:
            issues.append(_issue(f"l2.{axis_name}", f"L2 axis {axis_name} is not sweepable"))

    resolved = resolve_axes_from_raw(fixed_axes, leaf_config, l1_context=l1_context)
    issues.extend(_validate_options(fixed_axes, resolved))
    issues.extend(_validate_l2a_gates(fixed_axes, resolved, l1_context))
    issues.extend(_validate_transform(leaf_config, resolved, l1_context))
    issues.extend(_validate_sd_tcode_policy(leaf_config, resolved))
    issues.extend(_validate_outlier(leaf_config, resolved))
    issues.extend(_validate_imputation(leaf_config, resolved, l1_context))
    issues.extend(_validate_frame_edge(leaf_config, resolved))

    if resolved.get("imputation_policy") == "forward_fill":
        issues.append(
            Issue(
                "l2_forward_fill_warning",
                Severity.SOFT,
                "layer",
                "l2.imputation_policy",
                "forward-fill can introduce persistence artifacts; em_factor is standard",
            )
        )
    if resolved.get("frame_edge_policy") == "zero_fill_leading":
        issues.append(
            Issue(
                "l2_zero_fill_warning",
                Severity.SOFT,
                "layer",
                "l2.frame_edge_policy",
                "zero-fill can be learned as signal by ML models; truncate_to_balanced is recommended",
            )
        )
    return ValidationReport(tuple(issues))


def validate_recipe(recipe_yaml: dict[str, Any] | str) -> Any:
    from ..validator import ValidationReport

    root = parse_recipe_yaml(recipe_yaml) if isinstance(recipe_yaml, str) else recipe_yaml
    l1_context = _l1_context_from_recipe(root)
    report = ValidationReport()
    if "2_preprocessing" in root:
        report = report.extend(validate_layer(root["2_preprocessing"], l1_context=l1_context).issues)
    return report


def topological_order(dag: DAG) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in seen:
            return
        for ref in dag.nodes[node_id].inputs:
            if ref.node_id in dag.nodes:
                visit(ref.node_id)
        seen.add(node_id)
        result.append(node_id)

    for sink in dag.sinks.values():
        visit(sink)
    return result


def build_recipe_with_l2_only(yaml_text: str) -> L2Recipe:
    return L2Recipe(layer=parse_layer_yaml(yaml_text))


def execute_recipe(recipe: L2Recipe) -> L2Manifest:
    report = validate_layer(recipe.layer)
    if report.has_hard_errors:
        raise ValueError("; ".join(issue.message for issue in report.hard_errors))
    panel = Panel()
    artifact = L2CleanPanelArtifact(
        panel=panel,
        cleaning_log={"runtime": "schema_only"},
        cleaning_temporal_rules={"imputation": "expanding_window_per_origin"},
    )
    return L2Manifest(
        layer_execution_log={
            "l2": L2LayerExecutionRecord(
                layer_id="l2",
                status="completed",
                artifact=artifact,
                resolved_axes=dict(resolve_axes(normalize_to_dag_form(recipe.layer))),
            )
        }
    )


def _validate_options(fixed_axes: dict[str, Any], resolved: L2ResolvedAxes) -> list[Any]:
    option_sets = {
        "sd_series_frequency_filter": {"monthly_only", "quarterly_only", "both"},
        "mixed_frequency_representation": {
            "calendar_aligned_frame",
            "drop_unknown_native_frequency",
            "drop_non_target_native_frequency",
            "native_frequency_block_payload",
            "mixed_frequency_model_adapter",
        },
        "quarterly_to_monthly_rule": {"linear_interpolation", "step_backward", "step_forward", "chow_lin"},
        "monthly_to_quarterly_rule": {"quarterly_average", "quarterly_endpoint", "quarterly_sum"},
        "transform_policy": {"apply_official_tcode", "no_transform", "custom_tcode"},
        "sd_tcode_policy": {"none", "inferred", "empirical"},
        "transform_scope": {"target_and_predictors", "predictors_only", "target_only", "not_applicable"},
        "outlier_policy": {"mccracken_ng_iqr", "winsorize", "zscore_threshold", "none"},
        "outlier_action": {"flag_as_nan", "replace_with_median", "replace_with_cap_value", "keep_with_indicator"},
        "outlier_scope": {"target_and_predictors", "predictors_only", "target_only", "not_applicable"},
        "imputation_policy": {"em_factor", "em_multivariate", "mean", "forward_fill", "linear_interpolation", "none_propagate"},
        "imputation_temporal_rule": {"expanding_window_per_origin", "rolling_window_per_origin", "block_recompute"},
        "imputation_scope": {"target_and_predictors", "predictors_only", "target_only", "not_applicable"},
        "frame_edge_policy": {"truncate_to_balanced", "drop_unbalanced_series", "keep_unbalanced", "zero_fill_leading"},
        "frame_edge_scope": {"target_and_predictors", "predictors_only", "target_only", "not_applicable"},
    }
    issues = []
    for axis_name, allowed in option_sets.items():
        value = fixed_axes.get(axis_name, resolved.get(axis_name))
        if value is None or _is_sweep_marker(value):
            continue
        if value not in allowed:
            issues.append(_issue(f"l2.{axis_name}", f"{axis_name} must be one of {sorted(allowed)}"))
    return issues


def _validate_l2a_gates(fixed_axes: dict[str, Any], resolved: L2ResolvedAxes, l1_context: dict[str, Any] | None) -> list[Any]:
    issues = []
    if l1_context is None:
        return issues
    if not _l2a_active(l1_context):
        for axis in (
            "sd_series_frequency_filter",
            "mixed_frequency_representation",
            "quarterly_to_monthly_rule",
            "monthly_to_quarterly_rule",
            "sd_tcode_policy",
        ):
            if axis in fixed_axes:
                issues.append(_issue(f"l2.{axis}", f"{axis} is inactive when L1 dataset has no FRED-SD"))
        return issues
    frequency = l1_context.get("frequency")
    filter_value = resolved.get("sd_series_frequency_filter")
    q_to_m_active = frequency == "monthly" and filter_value in {"quarterly_only", "both"}
    m_to_q_active = frequency == "quarterly" and filter_value in {"monthly_only", "both"}
    if "quarterly_to_monthly_rule" in fixed_axes and not q_to_m_active:
        issues.append(_issue("l2.quarterly_to_monthly_rule", "quarterly_to_monthly_rule is inactive for this L1 frequency/filter"))
    if "monthly_to_quarterly_rule" in fixed_axes and not m_to_q_active:
        issues.append(_issue("l2.monthly_to_quarterly_rule", "monthly_to_quarterly_rule is inactive for this L1 frequency/filter"))
    if q_to_m_active and m_to_q_active:
        issues.append(_issue("l2.frequency_alignment", "quarterly_to_monthly_rule and monthly_to_quarterly_rule cannot both be active"))
    if resolved.get("quarterly_to_monthly_rule") == "chow_lin":
        issues.append(_issue("l2.quarterly_to_monthly_rule", "quarterly_to_monthly_rule=chow_lin is future and not executable"))
    return issues


def _validate_transform(leaf_config: dict[str, Any], resolved: L2ResolvedAxes, l1_context: dict[str, Any] | None) -> list[Any]:
    policy = resolved.get("transform_policy")
    if policy == "custom_tcode":
        mapping = leaf_config.get("custom_tcode_map")
        if not isinstance(mapping, dict) or any(not isinstance(v, int) or v < 1 or v > 7 for v in mapping.values()):
            return [_issue("l2.custom_tcode_map", "custom_tcode requires custom_tcode_map with tcode integers 1..7")]
    if policy == "apply_official_tcode" and l1_context and l1_context.get("custom_source_policy") == "custom_panel_only":
        if not l1_context.get("custom_has_tcode_column"):
            return [_issue("l2.transform_policy", "apply_official_tcode requires FRED-MD/QD metadata or documented custom tcode column")]
    return []


def _validate_sd_tcode_policy(leaf_config: dict[str, Any], resolved: L2ResolvedAxes) -> list[Any]:
    """Validate ``sd_tcode_policy`` axis + its leaf_config requirements.

    Schema added in v0.8.5: orthogonal to ``transform_policy``. Three options:
    ``none`` (default), ``inferred`` (national-analog research layer),
    ``empirical`` (variable-global / state-series stationarity audit map).
    The ``empirical`` mode requires a ``sd_tcode_unit`` leaf
    (``variable_global`` or ``state_series``) and, for ``state_series``,
    a ``sd_tcode_code_map`` and ``sd_tcode_audit_uri``.
    """

    policy = resolved.get("sd_tcode_policy")
    if policy != "empirical":
        return []
    issues: list[Any] = []
    unit = leaf_config.get("sd_tcode_unit")
    if unit not in {"variable_global", "state_series"}:
        issues.append(_issue(
            "l2.sd_tcode_unit",
            "sd_tcode_policy=empirical requires leaf_config.sd_tcode_unit "
            "in {variable_global, state_series}",
        ))
    if unit == "state_series":
        code_map = leaf_config.get("sd_tcode_code_map")
        if not isinstance(code_map, dict) or not code_map:
            issues.append(_issue(
                "l2.sd_tcode_code_map",
                "sd_tcode_policy=empirical with sd_tcode_unit=state_series "
                "requires non-empty leaf_config.sd_tcode_code_map",
            ))
    return issues


def _validate_outlier(leaf_config: dict[str, Any], resolved: L2ResolvedAxes) -> list[Any]:
    issues = []
    policy = resolved.get("outlier_policy")
    action = resolved.get("outlier_action")
    if policy == "mccracken_ng_iqr" and not _positive_number(leaf_config.get("outlier_iqr_threshold", 10.0)):
        issues.append(_issue("l2.outlier_iqr_threshold", "outlier_iqr_threshold must be positive"))
    if policy == "winsorize" and not _valid_quantiles(leaf_config.get("winsorize_quantiles", [0.01, 0.99])):
        issues.append(_issue("l2.winsorize_quantiles", "winsorize_quantiles must be two increasing probabilities"))
    if policy == "zscore_threshold" and not _positive_number(leaf_config.get("zscore_threshold_value", 3.0)):
        issues.append(_issue("l2.zscore_threshold_value", "zscore_threshold_value must be positive"))
    if action == "replace_with_cap_value" and policy != "winsorize":
        issues.append(_issue("l2.outlier_action", "replace_with_cap_value requires outlier_policy=winsorize"))
    if action == "keep_with_indicator":
        issues.append(_issue("l2.outlier_action", "outlier_action=keep_with_indicator is future and not executable"))
    return issues


def _validate_imputation(leaf_config: dict[str, Any], resolved: L2ResolvedAxes, l1_context: dict[str, Any] | None) -> list[Any]:
    issues = []
    policy = resolved.get("imputation_policy")
    temporal_rule = resolved.get("imputation_temporal_rule")
    if temporal_rule == "full_sample_once":
        issues.append(_issue("l2.imputation_temporal_rule", "full_sample_once is rejected because it leaks future data"))
    if policy == "em_factor":
        if not isinstance(leaf_config.get("em_n_factors", 8), int) or leaf_config.get("em_n_factors", 8) <= 0:
            issues.append(_issue("l2.em_n_factors", "em_n_factors must be a positive integer"))
    if policy in {"em_factor", "em_multivariate"}:
        if not isinstance(leaf_config.get("em_max_iter", 50), int) or leaf_config.get("em_max_iter", 50) <= 0:
            issues.append(_issue("l2.em_max_iter", "em_max_iter must be a positive integer"))
        if not _positive_number(leaf_config.get("em_tolerance", 1e-3)):
            issues.append(_issue("l2.em_tolerance", "em_tolerance must be positive"))
    frequency = (l1_context or {}).get("frequency", "monthly")
    if temporal_rule == "rolling_window_per_origin":
        default = 240 if frequency == "monthly" else 60
        if not isinstance(leaf_config.get("rolling_window_size", default), int) or leaf_config.get("rolling_window_size", default) <= 0:
            issues.append(_issue("l2.rolling_window_size", "rolling_window_size must be a positive integer"))
    if temporal_rule == "block_recompute":
        default = 12 if frequency == "monthly" else 4
        if not isinstance(leaf_config.get("block_recompute_interval", default), int) or leaf_config.get("block_recompute_interval", default) <= 0:
            issues.append(_issue("l2.block_recompute_interval", "block_recompute_interval must be a positive integer"))
    return issues


def _validate_frame_edge(leaf_config: dict[str, Any], resolved: L2ResolvedAxes) -> list[Any]:
    value = leaf_config.get("min_observation_per_series", 0)
    if not isinstance(value, int) or value < 0:
        return [_issue("l2.min_observation_per_series", "min_observation_per_series must be a non-negative integer")]
    return []


def _derived_scope(axis_name: str, values: dict[str, Any]) -> str:
    if axis_name == "transform_scope":
        return "not_applicable" if values.get("transform_policy") == "no_transform" else "target_and_predictors"
    if axis_name == "outlier_scope":
        return "not_applicable" if values.get("outlier_policy") == "none" else "predictors_only"
    if axis_name == "imputation_scope":
        return "not_applicable" if values.get("imputation_policy") == "none_propagate" else "predictors_only"
    if axis_name == "frame_edge_scope":
        return "not_applicable" if values.get("frame_edge_policy") == "keep_unbalanced" else "predictors_only"
    raise KeyError(axis_name)


def _step_axes(step_id: str) -> tuple[str, ...]:
    return {
        "freq_alignment": ("sd_series_frequency_filter", "quarterly_to_monthly_rule", "monthly_to_quarterly_rule"),
        "transform": ("transform_policy", "transform_scope"),
        "outlier_handle": ("outlier_policy", "outlier_action", "outlier_scope"),
        "imputation": ("imputation_policy", "imputation_temporal_rule", "imputation_scope"),
        "frame_edge": ("frame_edge_policy", "frame_edge_scope"),
    }[step_id]


def _axis_gates(axis_name: str) -> tuple[GatePredicate, ...]:
    if axis_name == "quarterly_to_monthly_rule":
        return (
            GatePredicate(
                kind="combined",
                target="",
                value=(
                    {"kind": "layer_axis_equals", "target": "l1.frequency", "value": "monthly"},
                    {"kind": "axis_in", "target": "sd_series_frequency_filter", "value": ["quarterly_only", "both"]},
                ),
            ),
        )
    if axis_name == "monthly_to_quarterly_rule":
        return (
            GatePredicate(
                kind="combined",
                target="",
                value=(
                    {"kind": "layer_axis_equals", "target": "l1.frequency", "value": "quarterly"},
                    {"kind": "axis_in", "target": "sd_series_frequency_filter", "value": ["monthly_only", "both"]},
                ),
            ),
        )
    return ()


def _step_gates(step_id: str) -> tuple[GatePredicate, ...]:
    if step_id == "freq_alignment":
        return (GatePredicate(kind="layer_axis_equals", target="l1.has_fred_sd", value=True),)
    return ()


def _l1_context_from_recipe(root: dict[str, Any]) -> dict[str, Any]:
    l1 = root.get("1_data", {}) or {}
    fixed = l1.get("fixed_axes", {}) or {}
    dataset = fixed.get("dataset", "fred_md")
    frequency = fixed.get("frequency")
    if frequency is None:
        frequency = "monthly" if dataset in {"fred_md", "fred_md+fred_sd"} else "quarterly" if dataset in {"fred_qd", "fred_qd+fred_sd"} else None
    return {
        "dataset": dataset,
        "frequency": frequency,
        "has_fred_sd": dataset in FRED_SD_DATASETS,
        "custom_source_policy": fixed.get("custom_source_policy", "official_only"),
    }


def _l2a_active(l1_context: dict[str, Any] | None) -> bool:
    if l1_context is None:
        return True
    return bool(l1_context.get("has_fred_sd") or l1_context.get("dataset") in FRED_SD_DATASETS)


def _positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and value > 0


def _valid_quantiles(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(v, (int, float)) for v in value)
        and 0 <= value[0] < value[1] <= 1
    )


def _is_sweep_marker(value: Any) -> bool:
    return isinstance(value, dict) and "sweep" in value


def _raw_layer(layer: Any | dict[str, Any]) -> dict[str, Any]:
    return layer.raw_yaml if hasattr(layer, "raw_yaml") else layer


def _issue(location: str, message: str) -> Any:
    from ..validator import Issue, Severity

    return Issue("l2_contract", Severity.HARD, "layer", location, message)


def _options(values: tuple[str, ...], future: tuple[str, ...] = ()) -> tuple[Option, ...]:
    return tuple(Option(value, value.replace("_", " ").title(), "", status="future" if value in future else "operational") for value in values)


L2_LAYER_SPEC = LayerImplementationSpec(
    layer_id="l2",
    name="Preprocessing",
    category="construction",
    expected_inputs=("l1_data_definition_v1",),
    produces=("l2_clean_panel_v1",),
    ui_mode="list",
    layer_globals=(),
    sub_layers=(
        SubLayerSpec(id="l2_a", name="Mixed frequency alignment", axes=("sd_series_frequency_filter", "mixed_frequency_representation", "quarterly_to_monthly_rule", "monthly_to_quarterly_rule")),
        SubLayerSpec(id="l2_b", name="Transform", axes=("transform_policy", "sd_tcode_policy", "transform_scope")),
        SubLayerSpec(id="l2_c", name="Outlier handling", axes=("outlier_policy", "outlier_action", "outlier_scope")),
        SubLayerSpec(id="l2_d", name="Imputation", axes=("imputation_policy", "imputation_temporal_rule", "imputation_scope")),
        SubLayerSpec(id="l2_e", name="Frame edge", axes=("frame_edge_policy", "frame_edge_scope")),
    ),
    axes={
        "l2_a": {
            "sd_series_frequency_filter": AxisSpec("sd_series_frequency_filter", _options(("monthly_only", "quarterly_only", "both")), "both"),
            "mixed_frequency_representation": AxisSpec(
                "mixed_frequency_representation",
                _options((
                    "calendar_aligned_frame",
                    "drop_unknown_native_frequency",
                    "drop_non_target_native_frequency",
                    "native_frequency_block_payload",
                    "mixed_frequency_model_adapter",
                )),
                "calendar_aligned_frame",
            ),
            "quarterly_to_monthly_rule": AxisSpec("quarterly_to_monthly_rule", _options(("linear_interpolation", "step_backward", "step_forward", "chow_lin"), future=("chow_lin",)), "step_backward"),
            "monthly_to_quarterly_rule": AxisSpec("monthly_to_quarterly_rule", _options(("quarterly_average", "quarterly_endpoint", "quarterly_sum")), "quarterly_average"),
        },
        "l2_b": {
            "transform_policy": AxisSpec("transform_policy", _options(("apply_official_tcode", "no_transform", "custom_tcode")), "apply_official_tcode"),
            "sd_tcode_policy": AxisSpec("sd_tcode_policy", _options(("none", "inferred", "empirical")), "none"),
            "transform_scope": AxisSpec("transform_scope", _options(("target_and_predictors", "predictors_only", "target_only", "not_applicable")), "derived"),
        },
        "l2_c": {
            "outlier_policy": AxisSpec("outlier_policy", _options(("mccracken_ng_iqr", "winsorize", "zscore_threshold", "none")), "mccracken_ng_iqr"),
            "outlier_action": AxisSpec("outlier_action", _options(("flag_as_nan", "replace_with_median", "replace_with_cap_value", "keep_with_indicator"), future=("keep_with_indicator",)), "flag_as_nan"),
            "outlier_scope": AxisSpec("outlier_scope", _options(("target_and_predictors", "predictors_only", "target_only", "not_applicable")), "derived"),
        },
        "l2_d": {
            "imputation_policy": AxisSpec("imputation_policy", _options(("em_factor", "em_multivariate", "mean", "forward_fill", "linear_interpolation", "none_propagate")), "em_factor"),
            "imputation_temporal_rule": AxisSpec("imputation_temporal_rule", _options(("expanding_window_per_origin", "rolling_window_per_origin", "block_recompute")), "expanding_window_per_origin"),
            "imputation_scope": AxisSpec("imputation_scope", _options(("target_and_predictors", "predictors_only", "target_only", "not_applicable")), "derived"),
        },
        "l2_e": {
            "frame_edge_policy": AxisSpec("frame_edge_policy", _options(("truncate_to_balanced", "drop_unbalanced_series", "keep_unbalanced", "zero_fill_leading")), "truncate_to_balanced"),
            "frame_edge_scope": AxisSpec("frame_edge_scope", _options(("target_and_predictors", "predictors_only", "target_only", "not_applicable")), "derived"),
        },
    },
)
