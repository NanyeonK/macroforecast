from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..dag import DAG, Node, NodeRef, SourceSelector


@dataclass(frozen=True)
class SubLayerSpec:
    axes: tuple[str, ...]


class L2_5PrePostPreprocessing:
    """Layer 2.5 Pre vs Post Preprocessing diagnostic implementation marker."""

    sub_layers = {
        "L2_5_A_comparison_axis": SubLayerSpec(("comparison_pair", "comparison_output_form")),
        "L2_5_B_distribution_shift": SubLayerSpec(("distribution_metric", "distribution_view")),
        "L2_5_C_correlation_shift": SubLayerSpec(("correlation_shift", "correlation_method")),
        "L2_5_D_cleaning_effect_summary": SubLayerSpec(("cleaning_summary_view", "t_code_application_log")),
        "L2_5_Z_export": SubLayerSpec(("diagnostic_format", "attach_to_manifest", "figure_dpi", "latex_export")),
    }

    @classmethod
    def list_axes(cls) -> tuple[str, ...]:
        return tuple(axis for spec in cls.sub_layers.values() for axis in spec.axes)

    @classmethod
    def list_sublayers(cls) -> tuple[str, ...]:
        return tuple(cls.sub_layers)


class L2_5ResolvedAxes(dict):
    def __init__(self, values: dict[str, Any], active: dict[str, bool]) -> None:
        super().__init__(values)
        self._active = active

    def get_active(self, key: str) -> bool:
        return self._active.get(key, True)


AXIS_NAMES = L2_5PrePostPreprocessing.list_axes()

DEFAULT_AXES: dict[str, Any] = {
    "comparison_pair": "raw_vs_final_clean",
    "comparison_output_form": "multi",
    "distribution_metric": ["mean_change", "sd_change", "ks_statistic"],
    "distribution_view": "multi",
    "correlation_shift": "none",
    "correlation_method": "pearson",
    "cleaning_summary_view": "multi",
    "t_code_application_log": "summary",
    "diagnostic_format": "pdf",
    "attach_to_manifest": True,
    "figure_dpi": 300,
    "latex_export": True,
}

OPTIONS = {
    "comparison_pair": {"raw_vs_final_clean", "raw_vs_tcoded", "raw_vs_outlier_handled", "raw_vs_imputed", "multi_stage"},
    "comparison_output_form": {"side_by_side_summary", "overlay_timeseries", "difference_table", "distribution_shift", "multi"},
    "distribution_view": {"summary_table", "qq_plot", "histogram_overlay", "multi"},
    "correlation_shift": {"none", "delta_matrix", "pre_post_overlay"},
    "correlation_method": {"pearson", "spearman"},
    "cleaning_summary_view": {"n_imputed_per_series", "n_outliers_flagged", "n_truncated_obs", "multi"},
    "t_code_application_log": {"none", "summary", "per_series_detail"},
    "diagnostic_format": {"png", "pdf", "html", "json", "latex_table", "csv", "multi"},
}
DISTRIBUTION_METRICS = {"mean_change", "sd_change", "skew_change", "kurtosis_change", "ks_statistic"}


def parse_layer_yaml(yaml_text: str, layer_id: Literal["l2_5"] = "l2_5") -> dict[str, Any]:
    if layer_id != "l2_5":
        raise ValueError("L2.5 parser only accepts layer_id='l2_5'")
    from ..yaml import parse_recipe_yaml

    root = parse_recipe_yaml(yaml_text)
    raw = root.get("2_5_pre_post_preprocessing", root)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("2_5_pre_post_preprocessing: layer YAML must be a mapping")
    return raw


def parse_recipe_yaml(yaml_text_or_root: str | dict[str, Any]) -> Any:
    from ..yaml import parse_recipe_yaml as parse

    root = parse(yaml_text_or_root) if isinstance(yaml_text_or_root, str) else yaml_text_or_root
    return L2_5Recipe(root)


@dataclass(frozen=True)
class L2_5Layer:
    raw_yaml: dict[str, Any]
    dag: DAG
    enabled: bool = False


@dataclass(frozen=True)
class L2_5Recipe:
    root: dict[str, Any]

    @property
    def layers(self) -> dict[str, Any]:
        if "2_5_pre_post_preprocessing" not in self.root:
            return {}
        raw = self.root["2_5_pre_post_preprocessing"] or {}
        enabled = bool(raw.get("enabled", False))
        return {"l2_5": L2_5Layer(raw, normalize_to_dag_form(raw, "l2_5"), enabled)}


def normalize_to_dag_form(layer: dict[str, Any] | L2_5Layer, layer_id: Literal["l2_5"] = "l2_5", context: dict[str, Any] | None = None) -> DAG:
    raw = layer.raw_yaml if isinstance(layer, L2_5Layer) else layer
    resolved = resolve_axes_from_raw(raw)
    if not resolved["enabled"]:
        return DAG("l2_5", {}, sinks={}, layer_globals={"resolved_axes": resolved})
    nodes: dict[str, Node] = {
        "src_l1_data": Node("src_l1_data", "source", "l2_5", "source", selector=SourceSelector("l1", "l1_data_definition_v1")),
        "src_l2_clean": Node("src_l2_clean", "source", "l2_5", "source", selector=SourceSelector("l2", "l2_clean_panel_v1")),
    }
    for axis in AXIS_NAMES:
        if resolved.get_active(axis):
            nodes[f"axis_{axis}"] = Node(f"axis_{axis}", "axis", "l2_5", axis, params={"value": resolved[axis]})
    previous = "step:collect_l2"
    nodes[previous] = Node(previous, "step", "l2_5", "diagnostic_collect_l2", inputs=(NodeRef("src_l1_data"), NodeRef("src_l2_clean")))
    for step in ("comparison_axis", "distribution_shift", "correlation_shift", "cleaning_effect_summary", "diagnostic_export"):
        node_id = f"step:{step}"
        axis_inputs = tuple(NodeRef(f"axis_{axis}") for axis in _step_axes(step) if f"axis_{axis}" in nodes)
        nodes[node_id] = Node(node_id, "step", "l2_5", f"l2_5_{step}", inputs=(NodeRef(previous),) + axis_inputs)
        previous = node_id
    nodes["sink:l2_5_diagnostic_v1"] = Node("sink:l2_5_diagnostic_v1", "sink", "l2_5", "sink", inputs=(NodeRef(previous),))
    return DAG("l2_5", nodes, sinks={"l2_5_diagnostic_v1": "sink:l2_5_diagnostic_v1"}, layer_globals={"resolved_axes": resolved})


def resolve_axes(dag_or_layer: DAG | L2_5Layer | dict[str, Any]) -> L2_5ResolvedAxes:
    if isinstance(dag_or_layer, DAG):
        return dag_or_layer.layer_globals.get("resolved_axes", resolve_axes_from_raw({}))
    raw = dag_or_layer.raw_yaml if isinstance(dag_or_layer, L2_5Layer) else dag_or_layer
    return resolve_axes_from_raw(raw)


def resolve_axes_from_raw(raw: dict[str, Any]) -> L2_5ResolvedAxes:
    fixed = raw.get("fixed_axes", {}) or {}
    values = {axis: _copy_default(default) for axis, default in DEFAULT_AXES.items()}
    values.update(fixed)
    values["enabled"] = bool(raw.get("enabled", False))
    values["leaf_config"] = raw.get("leaf_config", {}) or {}
    active = {axis: True for axis in AXIS_NAMES}
    if not values["enabled"]:
        active = {axis: False for axis in AXIS_NAMES}
    if values["correlation_shift"] == "none":
        active["correlation_method"] = False
    return L2_5ResolvedAxes(values, active)


def validate_layer(layer: dict[str, Any] | str, context: dict[str, Any] | None = None):
    from ..validator import ValidationReport

    raw = parse_layer_yaml(layer) if isinstance(layer, str) else layer
    fixed = raw.get("fixed_axes", {}) or {}
    resolved = resolve_axes_from_raw(raw)
    issues: list[Any] = []
    for axis, value in fixed.items():
        if axis not in AXIS_NAMES:
            issues.append(_issue(f"l2_5.{axis}", f"unknown L2.5 axis {axis!r}"))
        if _is_sweep(value):
            issues.append(_issue(f"l2_5.{axis}", "Diagnostic axes are not sweepable"))
    issues.extend(_validate_values(resolved))
    return ValidationReport(tuple(issues))


def validate_recipe(recipe: L2_5Recipe | dict[str, Any] | str):
    from ..validator import ValidationReport

    obj = parse_recipe_yaml(recipe) if isinstance(recipe, (str, dict)) else recipe
    if "2_5_pre_post_preprocessing" not in obj.root:
        return ValidationReport()
    return validate_layer(obj.root["2_5_pre_post_preprocessing"] or {})


def _validate_values(resolved: L2_5ResolvedAxes) -> list[Any]:
    issues: list[Any] = []
    if not resolved["enabled"]:
        return issues
    for axis, options in OPTIONS.items():
        if resolved.get_active(axis) and isinstance(resolved[axis], str) and resolved[axis] not in options:
            issues.append(_issue(f"l2_5.{axis}", f"invalid value {resolved[axis]!r} for {axis}"))
    metrics = resolved["distribution_metric"]
    if not isinstance(metrics, list) or any(metric not in DISTRIBUTION_METRICS for metric in metrics):
        issues.append(_issue("l2_5.distribution_metric", "distribution_metric entries must be valid L2.5 metrics"))
    return issues


def _step_axes(step: str) -> tuple[str, ...]:
    return {
        "comparison_axis": ("comparison_pair", "comparison_output_form"),
        "distribution_shift": ("distribution_metric", "distribution_view"),
        "correlation_shift": ("correlation_shift", "correlation_method"),
        "cleaning_effect_summary": ("cleaning_summary_view", "t_code_application_log"),
        "diagnostic_export": ("diagnostic_format", "attach_to_manifest", "figure_dpi", "latex_export"),
    }[step]


def _is_sweep(value: Any) -> bool:
    return isinstance(value, dict) and "sweep" in value


def _copy_default(value: Any) -> Any:
    return list(value) if isinstance(value, list) else value


def _issue(path: str, message: str) -> Any:
    from ..validator import Issue, Severity

    return Issue("l2_5_contract", Severity.HARD, "layer", path, message)


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
    """Build canonical AxisSpec from DEFAULT_AXES + OPTIONS (with diagnostic special cases)."""
    if name == "distribution_metric":
        opts = tuple(_opt(v) for v in sorted(DISTRIBUTION_METRICS))
    elif name == "attach_to_manifest":
        opts = (_Option("true", "True", ""), _Option("false", "False", ""))
    elif name in ("figure_dpi",):
        opts = ()
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
    "L2_5_A_comparison_axis": "Comparison axis",
    "L2_5_B_distribution_shift": "Distribution shift",
    "L2_5_C_correlation_shift": "Correlation shift",
    "L2_5_D_cleaning_effect_summary": "Cleaning effect summary",
    "L2_5_Z_export": "Diagnostic export",
}


L2_5_LAYER_SPEC = _LayerImplSpec(
    layer_id="l2_5",
    name="Pre vs post preprocessing",
    category="diagnostic",
    expected_inputs=("l1_data_definition_v1", "l2_clean_panel_v1"),
    produces=("l2_5_diagnostic_v1",),
    ui_mode="list",
    layer_globals=("enabled",),
    sub_layers=tuple(
        _CanonicalSubLayerSpec(id=sl_id, name=_SUBLAYER_NAMES[sl_id], axes=spec.axes)
        for sl_id, spec in L2_5PrePostPreprocessing.sub_layers.items()
    ),
    axes={
        sl_id: {axis: _build_axis(axis) for axis in spec.axes}
        for sl_id, spec in L2_5PrePostPreprocessing.sub_layers.items()
    },
)
