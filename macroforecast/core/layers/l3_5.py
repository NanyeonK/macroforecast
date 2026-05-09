from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..dag import DAG, Node, NodeRef, SourceSelector


@dataclass(frozen=True)
class SubLayerSpec:
    axes: tuple[str, ...]


class L3_5FeatureDiagnostics:
    """Layer 3.5 Feature Diagnostics implementation marker."""

    sub_layers = {
        "L3_5_A_comparison_axis": SubLayerSpec(("comparison_stages", "comparison_output_form")),
        "L3_5_B_factor_block_inspection": SubLayerSpec(("factor_view", "dfm_diagnostics")),
        "L3_5_C_feature_correlation": SubLayerSpec(("feature_correlation", "correlation_method", "correlation_view")),
        "L3_5_D_lag_block_inspection": SubLayerSpec(("lag_view", "marx_view")),
        "L3_5_E_selected_features_post_selection": SubLayerSpec(("selection_view", "stability_metric")),
        "L3_5_Z_export": SubLayerSpec(("diagnostic_format", "attach_to_manifest", "figure_dpi", "latex_export")),
    }

    @classmethod
    def list_axes(cls) -> tuple[str, ...]:
        return tuple(axis for spec in cls.sub_layers.values() for axis in spec.axes)

    @classmethod
    def list_sublayers(cls) -> tuple[str, ...]:
        return tuple(cls.sub_layers)


class L3_5ResolvedAxes(dict):
    def __init__(self, values: dict[str, Any], active: dict[str, bool]) -> None:
        super().__init__(values)
        self._active = active

    def get_active(self, key: str) -> bool:
        return self._active.get(key, True)


AXIS_NAMES = L3_5FeatureDiagnostics.list_axes()

DEFAULT_AXES: dict[str, Any] = {
    "comparison_stages": "cleaned_vs_features",
    "comparison_output_form": "multi",
    "factor_view": "multi",
    "dfm_diagnostics": "multi",
    "feature_correlation": "cross_block",
    "correlation_method": "pearson",
    "correlation_view": "clustered_heatmap",
    "lag_view": "multi",
    "marx_view": "weight_decay_visualization",
    "selection_view": "multi",
    "stability_metric": "jaccard",
    "diagnostic_format": "pdf",
    "attach_to_manifest": True,
    "figure_dpi": 300,
    "latex_export": True,
}

OPTIONS = {
    "comparison_stages": {"cleaned_vs_features", "raw_vs_cleaned_vs_features", "features_only"},
    "comparison_output_form": {"side_by_side", "dimension_summary", "distribution_shift", "multi"},
    "factor_view": {"scree_plot", "cumulative_variance", "loadings_heatmap", "factor_timeseries", "multi"},
    "dfm_diagnostics": {"none", "idiosyncratic_acf", "factor_var_stability", "multi"},
    "feature_correlation": {"none", "within_block", "cross_block", "with_target", "multi"},
    "correlation_method": {"pearson", "spearman"},
    "correlation_view": {"full_matrix", "clustered_heatmap", "top_k"},
    "lag_view": {"autocorrelation_per_lag", "partial_autocorrelation", "lag_correlation_decay", "multi"},
    "marx_view": {"none", "weight_decay_visualization"},
    "selection_view": {"selected_list", "selection_count_per_origin", "selection_stability", "multi"},
    "stability_metric": {"jaccard", "kuncheva"},
    "diagnostic_format": {"png", "pdf", "html", "json", "latex_table", "csv", "multi"},
}

FACTOR_OPS = {"pca", "sparse_pca", "scaled_pca", "dfm", "varimax", "varimax_rotation", "partial_least_squares", "random_projection"}
LAG_OPS = {"lag", "seasonal_lag", "ma_increasing_order"}


def parse_layer_yaml(yaml_text: str, layer_id: Literal["l3_5"] = "l3_5") -> dict[str, Any]:
    if layer_id != "l3_5":
        raise ValueError("L3.5 parser only accepts layer_id='l3_5'")
    from ..yaml import parse_recipe_yaml

    root = parse_recipe_yaml(yaml_text)
    raw = root.get("3_5_feature_diagnostics", root)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("3_5_feature_diagnostics: layer YAML must be a mapping")
    return raw


def parse_recipe_yaml(yaml_text_or_root: str | dict[str, Any]) -> Any:
    from ..yaml import parse_recipe_yaml as parse

    root = parse(yaml_text_or_root) if isinstance(yaml_text_or_root, str) else yaml_text_or_root
    return L3_5Recipe(root)


@dataclass(frozen=True)
class L3_5Layer:
    raw_yaml: dict[str, Any]
    dag: DAG
    enabled: bool = False


@dataclass(frozen=True)
class L3_5Recipe:
    root: dict[str, Any]

    @property
    def layers(self) -> dict[str, Any]:
        if "3_5_feature_diagnostics" not in self.root:
            return {}
        raw = self.root["3_5_feature_diagnostics"] or {}
        enabled = bool(raw.get("enabled", False))
        return {"l3_5": L3_5Layer(raw, normalize_to_dag_form(raw, "l3_5", context=_recipe_context(self.root)), enabled)}


def normalize_to_dag_form(layer: dict[str, Any] | L3_5Layer, layer_id: Literal["l3_5"] = "l3_5", context: dict[str, Any] | None = None) -> DAG:
    raw = layer.raw_yaml if isinstance(layer, L3_5Layer) else layer
    resolved = resolve_axes_from_raw(raw, context=context)
    if not resolved["enabled"]:
        return DAG("l3_5", {}, sinks={}, layer_globals={"resolved_axes": resolved})
    nodes: dict[str, Node] = {
        "src_l1_data": Node("src_l1_data", "source", "l3_5", "source", selector=SourceSelector("l1", "l1_data_definition_v1")),
        "src_l2_clean": Node("src_l2_clean", "source", "l3_5", "source", selector=SourceSelector("l2", "l2_clean_panel_v1")),
        "src_l3_features": Node("src_l3_features", "source", "l3_5", "source", selector=SourceSelector("l3", "l3_features_v1")),
        "src_l3_metadata": Node("src_l3_metadata", "source", "l3_5", "source", selector=SourceSelector("l3", "l3_metadata_v1")),
    }
    for axis in AXIS_NAMES:
        if resolved.get_active(axis):
            nodes[f"axis_{axis}"] = Node(f"axis_{axis}", "axis", "l3_5", axis, params={"value": resolved[axis]})
    previous = "step:collect_l3"
    nodes[previous] = Node(
        previous,
        "step",
        "l3_5",
        "diagnostic_collect_l3",
        inputs=(NodeRef("src_l1_data"), NodeRef("src_l2_clean"), NodeRef("src_l3_features"), NodeRef("src_l3_metadata")),
    )
    for step in ("comparison_axis", "factor_block_inspection", "feature_correlation", "lag_block_inspection", "selected_features", "diagnostic_export"):
        node_id = f"step:{step}"
        axis_inputs = tuple(NodeRef(f"axis_{axis}") for axis in _step_axes(step) if f"axis_{axis}" in nodes)
        nodes[node_id] = Node(node_id, "step", "l3_5", f"l3_5_{step}", inputs=(NodeRef(previous),) + axis_inputs)
        previous = node_id
    nodes["sink:l3_5_diagnostic_v1"] = Node("sink:l3_5_diagnostic_v1", "sink", "l3_5", "sink", inputs=(NodeRef(previous),))
    return DAG("l3_5", nodes, sinks={"l3_5_diagnostic_v1": "sink:l3_5_diagnostic_v1"}, layer_globals={"resolved_axes": resolved})


def resolve_axes(dag_or_layer: DAG | L3_5Layer | dict[str, Any]) -> L3_5ResolvedAxes:
    if isinstance(dag_or_layer, DAG):
        return dag_or_layer.layer_globals.get("resolved_axes", resolve_axes_from_raw({}))
    raw = dag_or_layer.raw_yaml if isinstance(dag_or_layer, L3_5Layer) else dag_or_layer
    return resolve_axes_from_raw(raw)


def resolve_axes_from_raw(raw: dict[str, Any], context: dict[str, Any] | None = None) -> L3_5ResolvedAxes:
    fixed = raw.get("fixed_axes", {}) or {}
    leaf = raw.get("leaf_config", {}) or {}
    values = {axis: _copy_default(default) for axis, default in DEFAULT_AXES.items()}
    values.update(fixed)
    values["enabled"] = bool(raw.get("enabled", False))
    values["leaf_config"] = {
        "n_factors_to_show": leaf.get("n_factors_to_show", 8),
        "loading_top_k_per_factor": leaf.get("loading_top_k_per_factor", 10),
        "correlation_top_k": leaf.get("correlation_top_k", 30),
        **leaf,
    }
    active = {axis: True for axis in AXIS_NAMES}
    context = context or {}
    if not values["enabled"]:
        active = {axis: False for axis in AXIS_NAMES}
    if not context.get("has_factor_step", False):
        active["factor_view"] = False
    if not context.get("has_dfm_step", False):
        active["dfm_diagnostics"] = False
    if not context.get("has_lag_step", False):
        active["lag_view"] = False
    if not context.get("has_marx_step", False):
        active["marx_view"] = False
    if not context.get("has_feature_selection_step", False):
        active["selection_view"] = False
        active["stability_metric"] = False
    if values["feature_correlation"] == "none":
        active["correlation_method"] = False
        active["correlation_view"] = False
    if values["selection_view"] not in {"selection_stability", "multi"}:
        active["stability_metric"] = False
    return L3_5ResolvedAxes(values, active)


def validate_layer(layer: dict[str, Any] | str, context: dict[str, Any] | None = None):
    from ..validator import ValidationReport

    raw = parse_layer_yaml(layer) if isinstance(layer, str) else layer
    context = context or {}
    fixed = raw.get("fixed_axes", {}) or {}
    resolved = resolve_axes_from_raw(raw, context=context)
    issues: list[Any] = []
    for axis, value in fixed.items():
        if axis not in AXIS_NAMES:
            issues.append(_issue(f"l3_5.{axis}", f"unknown L3.5 axis {axis!r}"))
        if _is_sweep(value):
            issues.append(_issue(f"l3_5.{axis}", "Diagnostic axes are not sweepable"))
    issues.extend(_validate_values(resolved, fixed, context))
    return ValidationReport(tuple(issues))


def validate_recipe(recipe: L3_5Recipe | dict[str, Any] | str):
    from ..validator import ValidationReport

    obj = parse_recipe_yaml(recipe) if isinstance(recipe, (str, dict)) else recipe
    if "3_5_feature_diagnostics" not in obj.root:
        return ValidationReport()
    return validate_layer(obj.root["3_5_feature_diagnostics"] or {}, context=_recipe_context(obj.root))


def _validate_values(resolved: L3_5ResolvedAxes, fixed: dict[str, Any], context: dict[str, Any]) -> list[Any]:
    issues: list[Any] = []
    if not resolved["enabled"]:
        return issues
    for axis, options in OPTIONS.items():
        if resolved.get_active(axis) and isinstance(resolved[axis], str) and resolved[axis] not in options:
            issues.append(_issue(f"l3_5.{axis}", f"invalid value {resolved[axis]!r} for {axis}"))
    if "factor_view" in fixed and not context.get("has_factor_step", False):
        issues.append(_issue("l3_5.factor_view", "factor_view requires a factor reduction step in L3"))
    if "dfm_diagnostics" in fixed and not context.get("has_dfm_step", False):
        issues.append(_issue("l3_5.dfm_diagnostics", "dfm_diagnostics requires a dfm step in L3"))
    if "lag_view" in fixed and not context.get("has_lag_step", False):
        issues.append(_issue("l3_5.lag_view", "lag_view requires a lag step in L3"))
    if "marx_view" in fixed and not context.get("has_marx_step", False):
        issues.append(_issue("l3_5.marx_view", "marx_view requires a ma_increasing_order step in L3"))
    if "selection_view" in fixed and not context.get("has_feature_selection_step", False):
        issues.append(_issue("l3_5.selection_view", "selection_view requires a feature_selection step in L3"))
    return issues


def _recipe_context(root: dict[str, Any]) -> dict[str, Any]:
    ops = _ops_from_nodes(((root.get("3_feature_engineering") or {}).get("nodes") or []))
    return {
        "has_factor_step": bool(ops & FACTOR_OPS),
        "has_dfm_step": "dfm" in ops,
        "has_lag_step": bool(ops & LAG_OPS),
        "has_marx_step": "ma_increasing_order" in ops,
        "has_feature_selection_step": "feature_selection" in ops,
    }


def _ops_from_nodes(nodes: list[Any]) -> set[str]:
    ops: set[str] = set()
    for node in nodes:
        if isinstance(node, dict) and node.get("type") in {"step", "combine"} and node.get("op"):
            ops.add(str(node["op"]))
    return ops


def _step_axes(step: str) -> tuple[str, ...]:
    return {
        "comparison_axis": ("comparison_stages", "comparison_output_form"),
        "factor_block_inspection": ("factor_view", "dfm_diagnostics"),
        "feature_correlation": ("feature_correlation", "correlation_method", "correlation_view"),
        "lag_block_inspection": ("lag_view", "marx_view"),
        "selected_features": ("selection_view", "stability_metric"),
        "diagnostic_export": ("diagnostic_format", "attach_to_manifest", "figure_dpi", "latex_export"),
    }[step]


def _is_sweep(value: Any) -> bool:
    return isinstance(value, dict) and "sweep" in value


def _copy_default(value: Any) -> Any:
    return list(value) if isinstance(value, list) else value


def _issue(path: str, message: str) -> Any:
    from ..validator import Issue, Severity

    return Issue("l3_5_contract", Severity.HARD, "layer", path, message)


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
    label = value.replace("_", " ").title()
    return _Option(value=value, label=label, description="")


def _build_axis(name: str) -> _AxisSpec:
    opts: tuple[_Option, ...]
    if name == "attach_to_manifest":
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
    "L3_5_A_comparison_axis": "Comparison axis",
    "L3_5_B_factor_block_inspection": "Factor block inspection",
    "L3_5_C_feature_correlation": "Feature correlation",
    "L3_5_D_lag_block_inspection": "Lag block inspection",
    "L3_5_E_selected_features_post_selection": "Selected features post-selection",
    "L3_5_Z_export": "Diagnostic export",
}


L3_5_LAYER_SPEC = _LayerImplSpec(
    layer_id="l3_5",
    name="Feature diagnostics",
    category="diagnostic",
    expected_inputs=("l1_data_definition_v1", "l2_clean_panel_v1", "l3_features_v1", "l3_metadata_v1"),
    produces=("l3_5_diagnostic_v1",),
    ui_mode="list",
    layer_globals=("enabled",),
    sub_layers=tuple(
        _CanonicalSubLayerSpec(id=sl_id, name=_SUBLAYER_NAMES[sl_id], axes=spec.axes)
        for sl_id, spec in L3_5FeatureDiagnostics.sub_layers.items()
    ),
    axes={
        sl_id: {axis: _build_axis(axis) for axis in spec.axes}
        for sl_id, spec in L3_5FeatureDiagnostics.sub_layers.items()
    },
)
