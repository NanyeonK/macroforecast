from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..dag import DAG, Node, NodeRef, SourceSelector


@dataclass(frozen=True)
class SubLayerSpec:
    axes: tuple[str, ...]


class L1_5DataSummary:
    """Layer 1.5 Data Summary diagnostic implementation marker."""

    sub_layers = {
        "L1_5_A_sample_coverage": SubLayerSpec(("coverage_view",)),
        "L1_5_B_univariate_summary": SubLayerSpec(("summary_metrics", "summary_split")),
        "L1_5_C_stationarity_tests": SubLayerSpec(("stationarity_test", "stationarity_test_scope")),
        "L1_5_D_missing_outlier_audit": SubLayerSpec(("missing_view", "outlier_view")),
        "L1_5_E_correlation_pre_cleaning": SubLayerSpec(("correlation_method", "correlation_view")),
        "L1_5_Z_export": SubLayerSpec(("diagnostic_format", "attach_to_manifest", "figure_dpi", "latex_export")),
    }

    @classmethod
    def list_axes(cls) -> tuple[str, ...]:
        return tuple(axis for spec in cls.sub_layers.values() for axis in spec.axes)

    @classmethod
    def list_sublayers(cls) -> tuple[str, ...]:
        return tuple(cls.sub_layers)


class L1_5ResolvedAxes(dict):
    def __init__(self, values: dict[str, Any], active: dict[str, bool]) -> None:
        super().__init__(values)
        self._active = active

    def get_active(self, key: str) -> bool:
        return self._active.get(key, True)


AXIS_NAMES = L1_5DataSummary.list_axes()

DEFAULT_AXES: dict[str, Any] = {
    "coverage_view": "multi",
    "summary_metrics": ["mean", "sd", "min", "max", "n_missing"],
    "summary_split": "full_sample",
    "stationarity_test": "none",
    "stationarity_test_scope": "target_and_predictors",
    "missing_view": "multi",
    "outlier_view": "iqr_flag",
    "correlation_method": "pearson",
    "correlation_view": "none",
    "diagnostic_format": "pdf",
    "attach_to_manifest": True,
    "figure_dpi": 300,
    "latex_export": True,
}

OPTIONS = {
    "coverage_view": {"per_series_start_end", "panel_balance_matrix", "observation_count", "multi"},
    "summary_split": {"full_sample", "pre_oos_only", "per_decade", "per_regime"},
    "stationarity_test": {"none", "adf", "pp", "kpss", "multi"},
    "stationarity_test_scope": {"target_only", "predictors_only", "target_and_predictors"},
    "missing_view": {"heatmap", "per_series_count", "longest_gap", "multi"},
    "outlier_view": {"none", "zscore_flag", "iqr_flag", "multi"},
    "correlation_method": {"pearson", "spearman", "kendall"},
    "correlation_view": {"none", "full_matrix", "clustered_heatmap", "top_k_per_target"},
    "diagnostic_format": {"png", "pdf", "html", "json", "latex_table", "csv", "multi"},
}
SUMMARY_METRICS = {"mean", "sd", "min", "max", "skew", "kurtosis", "n_obs", "n_missing"}


def parse_layer_yaml(yaml_text: str, layer_id: Literal["l1_5"] = "l1_5") -> dict[str, Any]:
    if layer_id != "l1_5":
        raise ValueError("L1.5 parser only accepts layer_id='l1_5'")
    from ..yaml import parse_recipe_yaml

    root = parse_recipe_yaml(yaml_text)
    raw = root.get("1_5_data_summary", root)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("1_5_data_summary: layer YAML must be a mapping")
    return raw


def parse_recipe_yaml(yaml_text_or_root: str | dict[str, Any]) -> Any:
    from ..yaml import parse_recipe_yaml as parse

    root = parse(yaml_text_or_root) if isinstance(yaml_text_or_root, str) else yaml_text_or_root
    return L1_5Recipe(root)


@dataclass(frozen=True)
class L1_5Layer:
    raw_yaml: dict[str, Any]
    dag: DAG
    enabled: bool = False


@dataclass(frozen=True)
class L1_5Recipe:
    root: dict[str, Any]

    @property
    def layers(self) -> dict[str, Any]:
        if "1_5_data_summary" not in self.root:
            return {}
        raw = self.root["1_5_data_summary"] or {}
        enabled = bool(raw.get("enabled", False))
        return {"l1_5": L1_5Layer(raw, normalize_to_dag_form(raw, "l1_5", context=_recipe_context(self.root)), enabled)}


def normalize_to_dag_form(layer: dict[str, Any] | L1_5Layer, layer_id: Literal["l1_5"] = "l1_5", context: dict[str, Any] | None = None) -> DAG:
    raw = layer.raw_yaml if isinstance(layer, L1_5Layer) else layer
    resolved = resolve_axes_from_raw(raw, context=context)
    if not resolved["enabled"]:
        return DAG("l1_5", {}, sinks={}, layer_globals={"resolved_axes": resolved})
    nodes: dict[str, Node] = {
        "src_l1_data": Node("src_l1_data", "source", "l1_5", "source", selector=SourceSelector("l1", "l1_data_definition_v1")),
    }
    for axis in AXIS_NAMES:
        if resolved.get_active(axis):
            nodes[f"axis_{axis}"] = Node(f"axis_{axis}", "axis", "l1_5", axis, params={"value": resolved[axis]})
    previous = "step:collect_l1"
    nodes[previous] = Node(previous, "step", "l1_5", "diagnostic_collect_l1", inputs=(NodeRef("src_l1_data"),))
    for step in ("sample_coverage", "univariate_summary", "stationarity_tests", "missing_outlier_audit", "correlation_pre_cleaning", "diagnostic_export"):
        node_id = f"step:{step}"
        axis_inputs = tuple(NodeRef(f"axis_{axis}") for axis in _step_axes(step) if f"axis_{axis}" in nodes)
        nodes[node_id] = Node(node_id, "step", "l1_5", f"l1_5_{step}", inputs=(NodeRef(previous),) + axis_inputs)
        previous = node_id
    nodes["sink:l1_5_diagnostic_v1"] = Node("sink:l1_5_diagnostic_v1", "sink", "l1_5", "sink", inputs=(NodeRef(previous),))
    return DAG("l1_5", nodes, sinks={"l1_5_diagnostic_v1": "sink:l1_5_diagnostic_v1"}, layer_globals={"resolved_axes": resolved})


def resolve_axes(dag_or_layer: DAG | L1_5Layer | dict[str, Any]) -> L1_5ResolvedAxes:
    if isinstance(dag_or_layer, DAG):
        return dag_or_layer.layer_globals.get("resolved_axes", resolve_axes_from_raw({}))
    raw = dag_or_layer.raw_yaml if isinstance(dag_or_layer, L1_5Layer) else dag_or_layer
    return resolve_axes_from_raw(raw)


def resolve_axes_from_raw(raw: dict[str, Any], context: dict[str, Any] | None = None) -> L1_5ResolvedAxes:
    fixed = raw.get("fixed_axes", {}) or {}
    leaf = raw.get("leaf_config", {}) or {}
    values = {axis: _copy_default(default) for axis, default in DEFAULT_AXES.items()}
    values.update(fixed)
    values["enabled"] = bool(raw.get("enabled", False))
    values["leaf_config"] = {
        "adf_max_lag": leaf.get("adf_max_lag", "auto"),
        "kpss_trend": leaf.get("kpss_trend", "c"),
        "outlier_threshold_iqr": leaf.get("outlier_threshold_iqr", 10.0),
        "outlier_zscore_threshold": leaf.get("outlier_zscore_threshold", 3.0),
        "correlation_top_k": leaf.get("correlation_top_k", 20),
        **leaf,
    }
    active = {axis: True for axis in AXIS_NAMES}
    if not values["enabled"]:
        active = {axis: False for axis in AXIS_NAMES}
    if values["stationarity_test"] == "none":
        active["stationarity_test_scope"] = False
    return L1_5ResolvedAxes(values, active)


def validate_layer(layer: dict[str, Any] | str, context: dict[str, Any] | None = None):
    from ..validator import ValidationReport

    raw = parse_layer_yaml(layer) if isinstance(layer, str) else layer
    context = context or {}
    fixed = raw.get("fixed_axes", {}) or {}
    resolved = resolve_axes_from_raw(raw, context=context)
    issues: list[Any] = []
    for axis, value in fixed.items():
        if axis not in AXIS_NAMES:
            issues.append(_issue(f"l1_5.{axis}", f"unknown L1.5 axis {axis!r}"))
        if _is_sweep(value):
            issues.append(_issue(f"l1_5.{axis}", "Diagnostic axes are not sweepable"))
    issues.extend(_validate_values(resolved, context))
    return ValidationReport(tuple(issues))


def validate_recipe(recipe: L1_5Recipe | dict[str, Any] | str):
    from ..validator import ValidationReport

    obj = parse_recipe_yaml(recipe) if isinstance(recipe, (str, dict)) else recipe
    if "1_5_data_summary" not in obj.root:
        return ValidationReport()
    return validate_layer(obj.root["1_5_data_summary"] or {}, context=_recipe_context(obj.root))


def _validate_values(resolved: L1_5ResolvedAxes, context: dict[str, Any]) -> list[Any]:
    issues: list[Any] = []
    if not resolved["enabled"]:
        return issues
    for axis, options in OPTIONS.items():
        if resolved.get_active(axis) and isinstance(resolved[axis], str) and resolved[axis] not in options:
            issues.append(_issue(f"l1_5.{axis}", f"invalid value {resolved[axis]!r} for {axis}"))
    metrics = resolved["summary_metrics"]
    if not isinstance(metrics, list) or any(metric not in SUMMARY_METRICS for metric in metrics):
        issues.append(_issue("l1_5.summary_metrics", "summary_metrics entries must be valid L1.5 metrics"))
    if resolved["summary_split"] == "per_regime" and not context.get("regime_active", False):
        issues.append(_issue("l1_5.summary_split", "summary_split=per_regime requires L1 regime_definition != none"))
    return issues


def _recipe_context(root: dict[str, Any]) -> dict[str, Any]:
    l1_axes = ((root.get("1_data") or {}).get("fixed_axes") or {})
    return {"regime_active": l1_axes.get("regime_definition", "none") != "none"}


def _step_axes(step: str) -> tuple[str, ...]:
    return {
        "sample_coverage": ("coverage_view",),
        "univariate_summary": ("summary_metrics", "summary_split"),
        "stationarity_tests": ("stationarity_test", "stationarity_test_scope"),
        "missing_outlier_audit": ("missing_view", "outlier_view"),
        "correlation_pre_cleaning": ("correlation_method", "correlation_view"),
        "diagnostic_export": ("diagnostic_format", "attach_to_manifest", "figure_dpi", "latex_export"),
    }[step]


def _is_sweep(value: Any) -> bool:
    return isinstance(value, dict) and "sweep" in value


def _copy_default(value: Any) -> Any:
    return list(value) if isinstance(value, list) else value


def _issue(path: str, message: str) -> Any:
    from ..validator import Issue, Severity

    return Issue("l1_5_contract", Severity.HARD, "layer", path, message)


# ---------------------------------------------------------------------------
# Canonical LAYER_SPEC (LayerImplementationSpec) — unified API per design
# ---------------------------------------------------------------------------

from ..layer_specs import (  # noqa: E402
    AxisSpec as _AxisSpec,
    LayerImplementationSpec as _LayerImplSpec,
    Option as _Option,
    SubLayerSpec as _CanonicalSubLayerSpec,
)


def _opt(value: str) -> _Option:
    """Lightweight Option factory: label = value (titlecased), description blank."""
    label = value.replace("_", " ").title()
    return _Option(value=value, label=label, description="")


def _build_axis(name: str) -> _AxisSpec:
    """Build canonical AxisSpec from DEFAULT_AXES + OPTIONS (and SUMMARY_METRICS for list axis)."""
    if name == "summary_metrics":
        # multi-select list axis
        opts = tuple(_opt(v) for v in sorted(SUMMARY_METRICS))
    elif name == "attach_to_manifest":
        opts = (_Option("true", "True", ""), _Option("false", "False", ""))
    elif name in ("figure_dpi",):
        opts = ()  # numeric, no enum
    elif name == "latex_export":
        opts = (_Option("true", "True", ""), _Option("false", "False", ""))
    else:
        opts = tuple(_opt(v) for v in sorted(OPTIONS.get(name, ()))) if name in OPTIONS else ()
    return _AxisSpec(
        name=name,
        options=opts,
        default=DEFAULT_AXES.get(name),
        sweepable=False,  # all diagnostic axes are non-sweepable (design Part 4)
    )


_SUBLAYER_NAMES = {
    "L1_5_A_sample_coverage": "Sample coverage",
    "L1_5_B_univariate_summary": "Univariate summary",
    "L1_5_C_stationarity_tests": "Stationarity tests",
    "L1_5_D_missing_outlier_audit": "Missing & outlier audit",
    "L1_5_E_correlation_pre_cleaning": "Correlation pre-cleaning",
    "L1_5_Z_export": "Diagnostic export",
}


L1_5_LAYER_SPEC = _LayerImplSpec(
    layer_id="l1_5",
    name="Data summary",
    category="diagnostic",
    expected_inputs=("l1_data_definition_v1",),
    produces=("l1_5_diagnostic_v1",),
    ui_mode="list",
    layer_globals=("enabled",),
    sub_layers=tuple(
        _CanonicalSubLayerSpec(id=sl_id, name=_SUBLAYER_NAMES[sl_id], axes=spec.axes)
        for sl_id, spec in L1_5DataSummary.sub_layers.items()
    ),
    axes={
        sl_id: {axis: _build_axis(axis) for axis in spec.axes}
        for sl_id, spec in L1_5DataSummary.sub_layers.items()
    },
)
