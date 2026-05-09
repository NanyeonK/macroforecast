from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..dag import DAG, Node, NodeRef, SourceSelector


class L5Evaluation:
    """Layer 5 Evaluation implementation marker."""

    @classmethod
    def list_axes(cls) -> tuple[str, ...]:
        return L5_AXIS_NAMES

    @classmethod
    def list_sublayers(cls) -> tuple[str, ...]:
        return ("L5.A", "L5.B", "L5.C", "L5.D", "L5.E")


L5_AXIS_NAMES: tuple[str, ...] = (
    "primary_metric",
    "point_metrics",
    "density_metrics",
    "direction_metrics",
    "relative_metrics",
    "benchmark_window",
    "benchmark_scope",
    "agg_time",
    "agg_horizon",
    "agg_target",
    "agg_state",
    "oos_period",
    "regime_use",
    "regime_metrics",
    "decomposition_target",
    "decomposition_order",
    "ranking",
    "report_style",
)

DEFAULT_AXES: dict[str, Any] = {
    "primary_metric": "mse",
    "point_metrics": ["mse", "mae"],
    "density_metrics": ["log_score", "crps"],
    "direction_metrics": [],
    "relative_metrics": ["relative_mse", "r2_oos"],
    "benchmark_window": "full_oos",
    "benchmark_scope": "all_targets_horizons",
    "agg_time": "mean",
    "agg_horizon": "per_horizon_separate",
    "agg_target": "per_target_separate",
    "agg_state": "pool_states",
    "oos_period": "full_oos",
    "regime_use": "pooled",
    "decomposition_target": "none",
    "decomposition_order": "marginal",
    "ranking": "by_primary_metric",
    "report_style": "single_table",
}

POINT_METRICS = {"mse", "rmse", "mae", "mape", "medae", "theil_u1", "theil_u2"}
DENSITY_METRICS = {"log_score", "crps", "interval_score", "coverage_rate"}
DIRECTION_METRICS = {"success_ratio", "pesaran_timmermann_metric"}
RELATIVE_METRICS = {"relative_mse", "r2_oos", "relative_mae", "mse_reduction"}


class L5ResolvedAxes(dict):
    def __init__(self, values: dict[str, Any], active: dict[str, bool]) -> None:
        super().__init__(values)
        self._active = active

    def get_active(self, key: str) -> bool:
        return self._active.get(key, True)


def parse_layer_yaml(yaml_text: str, layer_id: Literal["l5"] = "l5") -> dict[str, Any]:
    if layer_id != "l5":
        raise ValueError("L5 parser only accepts layer_id='l5'")
    from ..yaml import parse_recipe_yaml

    root = parse_recipe_yaml(yaml_text)
    raw = root.get("5_evaluation", root)
    if not isinstance(raw, dict):
        raise ValueError("5_evaluation: layer YAML must be a mapping")
    return raw


def parse_recipe_yaml(yaml_text_or_root: str | dict[str, Any]) -> Any:
    from ..yaml import parse_recipe_yaml as parse

    root = parse(yaml_text_or_root) if isinstance(yaml_text_or_root, str) else yaml_text_or_root
    return L5Recipe(root)


@dataclass(frozen=True)
class L5Layer:
    raw_yaml: dict[str, Any]
    dag: DAG


@dataclass(frozen=True)
class L5Recipe:
    root: dict[str, Any]

    @property
    def layers(self) -> dict[str, Any]:
        layers: dict[str, Any] = {}
        if "5_evaluation" in self.root:
            raw = self.root["5_evaluation"] or {}
            layers["l5"] = L5Layer(raw, normalize_to_dag_form(raw, "l5", context=_recipe_context(self.root)))
        return layers


def normalize_to_dag_form(layer: dict[str, Any] | L5Layer, layer_id: Literal["l5"] = "l5", context: dict[str, Any] | None = None) -> DAG:
    raw = layer.raw_yaml if isinstance(layer, L5Layer) else layer
    resolved = resolve_axes_from_raw(raw.get("fixed_axes", {}) or {}, context=context)
    nodes: dict[str, Node] = {
        "src_l4_forecasts": Node("src_l4_forecasts", "source", "l5", "source", selector=SourceSelector("l4", "l4_forecasts_v1")),
        "src_l4_model_artifacts": Node("src_l4_model_artifacts", "source", "l5", "source", selector=SourceSelector("l4", "l4_model_artifacts_v1")),
        "src_l1_data_definition": Node("src_l1_data_definition", "source", "l5", "source", selector=SourceSelector("l1", "l1_data_definition_v1")),
        "src_l1_regime_metadata": Node("src_l1_regime_metadata", "source", "l5", "source", selector=SourceSelector("l1", "l1_regime_metadata_v1")),
        "src_l3_metadata": Node("src_l3_metadata", "source", "l5", "source", selector=SourceSelector("l3", "l3_metadata_v1")),
    }
    for axis in L5_AXIS_NAMES:
        if not resolved.get_active(axis):
            continue
        nodes[f"axis_{axis}"] = Node("axis_" + axis, "axis", "l5", axis, params={"value": resolved[axis]})
    nodes["step:collect_inputs"] = Node(
        "step:collect_inputs",
        "step",
        "l5",
        "l5_collect_inputs",
        inputs=(NodeRef("src_l4_forecasts"), NodeRef("src_l4_model_artifacts"), NodeRef("src_l1_data_definition"), NodeRef("src_l1_regime_metadata"), NodeRef("src_l3_metadata")),
    )
    previous = "step:collect_inputs"
    for step in ("metric_compute", "benchmark_relative", "aggregate", "slice_and_decompose", "rank_and_report"):
        node_id = f"step:{step}"
        axis_inputs = tuple(NodeRef(f"axis_{axis}") for axis in _step_axes(step) if f"axis_{axis}" in nodes)
        nodes[node_id] = Node(node_id, "step", "l5", step, inputs=(NodeRef(previous),) + axis_inputs)
        previous = node_id
    nodes["sink:l5_evaluation_v1"] = Node("sink:l5_evaluation_v1", "sink", "l5", "sink", inputs=(NodeRef(previous),))
    return DAG("l5", nodes, sinks={"l5_evaluation_v1": "sink:l5_evaluation_v1"})


def resolve_axes(dag_or_layer: DAG | L5Layer | dict[str, Any]) -> L5ResolvedAxes:
    if isinstance(dag_or_layer, DAG):
        values = {node.id.removeprefix("axis_"): node.params["value"] for node in dag_or_layer.nodes.values() if node.id.startswith("axis_")}
        active = {axis: f"axis_{axis}" in dag_or_layer.nodes for axis in L5_AXIS_NAMES}
        return L5ResolvedAxes(values, active)
    raw = dag_or_layer.raw_yaml if isinstance(dag_or_layer, L5Layer) else dag_or_layer
    return resolve_axes_from_raw(raw.get("fixed_axes", {}) or {})


def resolve_axes_from_raw(fixed: dict[str, Any], context: dict[str, Any] | None = None) -> L5ResolvedAxes:
    values = {axis: _copy_default(DEFAULT_AXES[axis]) for axis in L5_AXIS_NAMES if axis in DEFAULT_AXES}
    values.update(fixed)
    active = {axis: True for axis in L5_AXIS_NAMES}
    context = context or {}
    if not context.get("has_benchmark", False):
        active["relative_metrics"] = False
        active["benchmark_window"] = False
        active["benchmark_scope"] = False
    if context.get("forecast_object", "point") not in {"quantile", "density"}:
        active["density_metrics"] = False
    if context.get("target_structure", "single_target") != "multi_series_target":
        active["agg_target"] = False
    if not context.get("has_fred_sd", False):
        active["agg_state"] = False
    if context.get("regime_definition", "none") == "none":
        active["regime_use"] = False
    if values.get("regime_use", "pooled") not in {"per_regime", "both"}:
        active["regime_metrics"] = False
    else:
        values.setdefault("regime_metrics", [values.get("primary_metric", "mse")])
    if values.get("decomposition_target") == "none":
        active["decomposition_order"] = False
    return L5ResolvedAxes(values, active)


def validate_layer(layer: dict[str, Any] | str, context: dict[str, Any] | None = None):
    from ..validator import Issue, Severity, ValidationReport

    raw = parse_layer_yaml(layer) if isinstance(layer, str) else layer
    fixed = raw.get("fixed_axes", {}) or {}
    leaf = raw.get("leaf_config", {}) or {}
    context = context or {}
    resolved = resolve_axes_from_raw(fixed, context=context)
    issues: list[Issue] = []
    for axis, value in fixed.items():
        if axis not in L5_AXIS_NAMES:
            issues.append(_issue(f"l5.{axis}", f"unknown L5 axis {axis!r}"))
        if _is_sweep(value):
            issues.append(_issue(f"l5.{axis}", "L5 axes are not sweepable"))
        if axis in L5_AXIS_NAMES and not resolved.get_active(axis):
            issues.append(_issue(f"l5.{axis}", f"{axis} is inactive for this recipe context"))
    issues.extend(_validate_metric_options(resolved, context))
    issues.extend(_validate_leaf_config(resolved, leaf))
    return ValidationReport(tuple(issues))


def validate_recipe(recipe: L5Recipe | dict[str, Any] | str):
    from ..validator import ValidationReport

    obj = parse_recipe_yaml(recipe) if isinstance(recipe, (str, dict)) else recipe
    root = obj.root
    if "5_evaluation" not in root:
        return ValidationReport()
    return validate_layer(root["5_evaluation"], context=_recipe_context(root))


def make_l5_yaml(primary_metric: str = "mse") -> str:
    return f"""
4_forecasting_model:
  nodes:
    - {{id: fit_a, type: step, op: fit_model, params: {{family: ridge}}, inputs: [src_X, src_y]}}
5_evaluation:
  fixed_axes:
    primary_metric: {primary_metric}
"""


def make_recipe_with_benchmark() -> L5Recipe:
    return parse_recipe_yaml(
        """
4_forecasting_model:
  nodes:
    - {id: fit_bench, type: step, op: fit_model, params: {family: ar_p}, is_benchmark: true, inputs: [src_y]}
5_evaluation:
  fixed_axes: {}
"""
    )


def make_recipe_without_benchmark() -> dict[str, Any]:
    return {
        "4_forecasting_model": {
            "nodes": [{"id": "fit_a", "type": "step", "op": "fit_model", "params": {"family": "ridge"}, "inputs": ["src_X", "src_y"]}]
        },
        "5_evaluation": {"fixed_axes": {}},
    }


def make_recipe_with_l3_metadata() -> dict[str, Any]:
    root = make_recipe_without_benchmark()
    root["3_feature_engineering"] = {"nodes": [], "sinks": {"l3_metadata_v1": "auto"}}
    return root


def _recipe_context(root: dict[str, Any]) -> dict[str, Any]:
    l1_fixed = ((root.get("1_data", {}) or {}).get("fixed_axes", {}) or {})
    l4_nodes = ((root.get("4_forecasting_model", {}) or {}).get("nodes", ()) or ())
    has_benchmark = any(node.get("is_benchmark") for node in l4_nodes if isinstance(node, dict))
    return {
        "has_benchmark": has_benchmark,
        "forecast_object": ((root.get("4_forecasting_model", {}) or {}).get("forecast_object", "point")),
        "target_structure": l1_fixed.get("target_structure", "single_target"),
        "has_fred_sd": l1_fixed.get("dataset", "fred_md") in {"fred_sd", "fred_md+fred_sd", "fred_qd+fred_sd"},
        "regime_definition": l1_fixed.get("regime_definition", "none"),
        "l6_mcs_active": bool((root.get("6_statistical_tests", {}) or {}).get("enabled", False)),
        "l3_metadata_available": "3_feature_engineering" in root or True,
    }


def _validate_metric_options(resolved: L5ResolvedAxes, context: dict[str, Any]) -> list[Any]:
    issues: list[Any] = []
    primary = resolved.get("primary_metric")
    if _is_sweep(primary):
        return issues
    if primary not in POINT_METRICS | RELATIVE_METRICS | {"log_score", "crps"}:
        issues.append(_issue("l5.primary_metric", "primary_metric must be a valid L5 metric"))
    if primary in {"relative_mse", "r2_oos"} and not context.get("has_benchmark", False):
        issues.append(_issue("l5.primary_metric", "relative metrics require an L4 benchmark"))
    if primary in {"log_score", "crps"} and context.get("forecast_object", "point") not in {"quantile", "density"}:
        issues.append(_issue("l5.primary_metric", "density metrics require quantile or density forecasts"))
    for metric in resolved.get("point_metrics", []):
        if metric not in POINT_METRICS:
            issues.append(_issue("l5.point_metrics", f"unknown point metric {metric!r}"))
    if resolved.get("ranking") == "by_relative_metric" and not context.get("has_benchmark", False):
        issues.append(_issue("l5.ranking", "by_relative_metric requires an L4 benchmark"))
    if resolved.get("ranking") == "mcs_inclusion" and not context.get("l6_mcs_active", False):
        issues.append(_issue("l5.ranking", "mcs_inclusion requires L6 MCS active"))
    if resolved.get("decomposition_target") == "by_state" and not context.get("has_fred_sd", False):
        issues.append(_issue("l5.decomposition_target", "by_state requires FRED-SD"))
    if resolved.get("decomposition_target") == "by_regime" and context.get("regime_definition", "none") == "none":
        issues.append(_issue("l5.decomposition_target", "by_regime requires active regime"))
    return issues


def _validate_leaf_config(resolved: L5ResolvedAxes, leaf: dict[str, Any]) -> list[Any]:
    issues = []
    if resolved.get("oos_period") == "fixed_dates" and not {"oos_start_date", "oos_end_date"} <= set(leaf):
        issues.append(_issue("l5.oos_period", "fixed_dates requires oos_start_date and oos_end_date"))
    if resolved.get("oos_period") == "multiple_subperiods" and "subperiod_list" not in leaf:
        issues.append(_issue("l5.oos_period", "multiple_subperiods requires subperiod_list"))
    if resolved.get("report_style") == "latex_table" and not {"latex_caption", "latex_label"} <= set(leaf):
        issues.append(_issue("l5.report_style", "latex_table requires latex_caption and latex_label"))
    if resolved.get("agg_time") == "per_subperiod" and "subperiod_dates" not in leaf:
        issues.append(_issue("l5.agg_time", "per_subperiod requires subperiod_dates"))
    return issues


def _step_axes(step: str) -> tuple[str, ...]:
    return {
        "metric_compute": ("primary_metric", "point_metrics", "density_metrics", "direction_metrics", "relative_metrics"),
        "benchmark_relative": ("benchmark_window", "benchmark_scope"),
        "aggregate": ("agg_time", "agg_horizon", "agg_target", "agg_state"),
        "slice_and_decompose": ("oos_period", "regime_use", "regime_metrics", "decomposition_target", "decomposition_order"),
        "rank_and_report": ("ranking", "report_style"),
    }[step]


def _copy_default(value: Any) -> Any:
    return list(value) if isinstance(value, list) else value


def _is_sweep(value: Any) -> bool:
    return isinstance(value, dict) and "sweep" in value


def _issue(location: str, message: str):
    from ..validator import Issue, Severity

    return Issue("l5_contract", Severity.HARD, "layer", location, message)


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


_AXIS_OPTION_SETS = {
    "primary_metric": POINT_METRICS | RELATIVE_METRICS | {"log_score", "crps"},
    "point_metrics": POINT_METRICS,
    "density_metrics": DENSITY_METRICS,
    "direction_metrics": DIRECTION_METRICS,
    "relative_metrics": RELATIVE_METRICS,
}


def _build_axis(name: str) -> _AxisSpec:
    opts = tuple(_opt(v) for v in sorted(_AXIS_OPTION_SETS.get(name, ())))
    return _AxisSpec(
        name=name,
        options=opts,
        default=DEFAULT_AXES.get(name),
        sweepable=False,
    )


# L5 sub-layers map to the 5 pipeline steps in _step_axes()
_L5_SUBLAYERS = (
    ("L5_A_metric_specification", "Metric specification", "metric_compute"),
    ("L5_B_benchmark_comparison", "Benchmark comparison", "benchmark_relative"),
    ("L5_C_aggregation", "Aggregation", "aggregate"),
    ("L5_D_sample_slicing_decomposition", "Sample slicing & decomposition", "slice_and_decompose"),
    ("L5_E_ranking_reporting", "Ranking & reporting", "rank_and_report"),
)


L5_LAYER_SPEC = _LayerImplSpec(
    layer_id="l5",
    name="Evaluation",
    category="consumption",
    expected_inputs=(
        "l4_forecasts_v1",
        "l4_model_artifacts_v1",
        "l1_data_definition_v1",
        "l1_regime_metadata_v1",
        "l3_metadata_v1",
    ),
    produces=("l5_evaluation_v1",),
    ui_mode="list",
    layer_globals=(),
    sub_layers=tuple(
        _CanonicalSubLayerSpec(id=sl_id, name=name, axes=_step_axes(step))
        for sl_id, name, step in _L5_SUBLAYERS
    ),
    axes={
        sl_id: {axis: _build_axis(axis) for axis in _step_axes(step)}
        for sl_id, _name, step in _L5_SUBLAYERS
    },
)
