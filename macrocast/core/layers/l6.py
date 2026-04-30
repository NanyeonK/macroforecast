from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..dag import DAG, Node, NodeRef, SourceSelector


class AxisGroup(dict):
    def __init__(self, values: dict[str, Any], active: dict[str, bool]) -> None:
        super().__init__(values)
        self._active = active

    def get_active(self, key: str) -> bool:
        return self._active.get(key, True)


class L6ResolvedAxes(dict):
    def __init__(self, values: dict[str, Any], active: dict[str, bool]) -> None:
        super().__init__(values)
        self._active = active

    def get_active(self, key: str) -> bool:
        return self._active.get(key, True)


@dataclass(frozen=True)
class SubLayerSpec:
    axes: tuple[str, ...]


class L6StatisticalTests:
    """Layer 6 Statistical Tests implementation marker."""

    layer_globals = ("test_scope", "dependence_correction", "overlap_handling")
    sub_layers = {
        "L6_A_equal_predictive": SubLayerSpec(("equal_predictive_test", "loss_function", "model_pair_strategy", "hln_correction")),
        "L6_B_nested": SubLayerSpec(("nested_test", "nested_pair_strategy", "cw_adjustment", "enc_test_one_sided")),
        "L6_C_cpa": SubLayerSpec(("cpa_test", "cpa_window_type", "cpa_conditioning_info", "cpa_critical_value_method")),
        "L6_D_multiple_model": SubLayerSpec(
            (
                "multiple_model_test",
                "mcs_alpha",
                "mmt_loss_function",
                "bootstrap_method",
                "bootstrap_n_replications",
                "bootstrap_block_length",
                "mcs_t_statistic",
                "spa_studentization",
                "stepm_alpha",
            )
        ),
        "L6_E_density_interval": SubLayerSpec(("density_test", "interval_test", "coverage_levels", "pit_n_bins", "pit_test_horizon_dependence")),
        "L6_F_direction": SubLayerSpec(("direction_test", "direction_threshold", "direction_alpha")),
        "L6_G_residual": SubLayerSpec(("residual_test", "residual_lag_count", "residual_test_scope", "residual_alpha")),
    }

    @classmethod
    def list_axes(cls) -> tuple[str, ...]:
        axes: list[str] = ["enabled", *cls.layer_globals]
        for spec in cls.sub_layers.values():
            axes.extend(spec.axes)
        return tuple(axes)

    @classmethod
    def list_sublayers(cls) -> tuple[str, ...]:
        return tuple(cls.sub_layers)


SUB_LAYER_ORDER = tuple(L6StatisticalTests.sub_layers)
SUB_LAYER_AXES = {name: spec.axes for name, spec in L6StatisticalTests.sub_layers.items()}
L6_AXIS_NAMES = L6StatisticalTests.list_axes()

GLOBAL_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "test_scope": "per_target_horizon",
    "dependence_correction": "newey_west",
    "overlap_handling": "nw_with_h_minus_1_lag",
}

SUB_LAYER_DEFAULTS: dict[str, dict[str, Any]] = {
    "L6_A_equal_predictive": {
        "equal_predictive_test": "dm_diebold_mariano",
        "loss_function": "squared",
        "model_pair_strategy": "vs_benchmark_only",
        "hln_correction": True,
    },
    "L6_B_nested": {
        "nested_test": "clark_west",
        "nested_pair_strategy": "vs_benchmark_auto",
        "cw_adjustment": True,
        "enc_test_one_sided": "one_sided",
    },
    "L6_C_cpa": {
        "cpa_test": "giacomini_rossi_2010",
        "cpa_window_type": "rolling_window",
        "cpa_conditioning_info": "none",
        "cpa_critical_value_method": "simulated",
    },
    "L6_D_multiple_model": {
        "multiple_model_test": "mcs_hansen",
        "mcs_alpha": 0.10,
        "mmt_loss_function": "squared",
        "bootstrap_method": "stationary_bootstrap",
        "bootstrap_n_replications": 1000,
        "bootstrap_block_length": "auto",
        "mcs_t_statistic": "t_max",
        "spa_studentization": "consistent",
        "stepm_alpha": 0.10,
    },
    "L6_E_density_interval": {
        "density_test": "pit_berkowitz",
        "interval_test": "christoffersen_conditional_coverage",
        "coverage_levels": [0.5, 0.9, 0.95],
        "pit_n_bins": 10,
        "pit_test_horizon_dependence": "nw_correction",
    },
    "L6_F_direction": {
        "direction_test": "pesaran_timmermann_1992",
        "direction_threshold": "zero",
        "direction_alpha": 0.05,
    },
    "L6_G_residual": {
        "residual_test": ["ljung_box_q", "arch_lm", "jarque_bera_normality"],
        "residual_lag_count": "derived",
        "residual_test_scope": "per_model_target_horizon",
        "residual_alpha": 0.05,
    },
}


def parse_layer_yaml(yaml_text: str, layer_id: Literal["l6"] = "l6") -> dict[str, Any]:
    if layer_id != "l6":
        raise ValueError("L6 parser only accepts layer_id='l6'")
    from ..yaml import parse_recipe_yaml

    root = parse_recipe_yaml(yaml_text)
    raw = root.get("6_statistical_tests", root)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("6_statistical_tests: layer YAML must be a mapping")
    return raw


def parse_recipe_yaml(yaml_text_or_root: str | dict[str, Any]) -> Any:
    from ..yaml import parse_recipe_yaml as parse

    root = parse(yaml_text_or_root) if isinstance(yaml_text_or_root, str) else yaml_text_or_root
    return L6Recipe(root)


@dataclass(frozen=True)
class L6Layer:
    raw_yaml: dict[str, Any]
    dag: DAG


@dataclass(frozen=True)
class L6Recipe:
    root: dict[str, Any]

    @property
    def layers(self) -> dict[str, Any]:
        layers: dict[str, Any] = {}
        if "6_statistical_tests" in self.root:
            raw = self.root["6_statistical_tests"] or {}
            layers["l6"] = L6Layer(raw, normalize_to_dag_form(raw, "l6", context=_recipe_context(self.root)))
        return layers


def normalize_to_dag_form(layer: dict[str, Any] | L6Layer, layer_id: Literal["l6"] = "l6", context: dict[str, Any] | None = None) -> DAG:
    raw = layer.raw_yaml if isinstance(layer, L6Layer) else layer
    resolved = resolve_axes_from_raw(raw, context=context)
    nodes: dict[str, Node] = {
        "src_l4_forecasts": Node("src_l4_forecasts", "source", "l6", "source", selector=SourceSelector("l4", "l4_forecasts_v1")),
        "src_l4_model_artifacts": Node("src_l4_model_artifacts", "source", "l6", "source", selector=SourceSelector("l4", "l4_model_artifacts_v1")),
        "src_l5_evaluation": Node("src_l5_evaluation", "source", "l6", "source", selector=SourceSelector("l5", "l5_evaluation_v1")),
        "src_l1_data_definition": Node("src_l1_data_definition", "source", "l6", "source", selector=SourceSelector("l1", "l1_data_definition_v1")),
        "src_l1_regime_metadata": Node("src_l1_regime_metadata", "source", "l6", "source", selector=SourceSelector("l1", "l1_regime_metadata_v1")),
    }
    nodes["axis_enabled"] = Node("axis_enabled", "axis", "l6", "enabled", params={"value": resolved["enabled"]})
    if resolved["enabled"]:
        for axis in L6StatisticalTests.layer_globals:
            nodes[f"axis_{axis}"] = Node(f"axis_{axis}", "axis", "l6", axis, params={"value": resolved[axis]})
        for sub_name in SUB_LAYER_ORDER:
            sub = resolved[sub_name]
            if not sub["enabled"]:
                continue
            for axis in SUB_LAYER_AXES[sub_name]:
                if sub.get_active(axis):
                    node_id = f"axis_{sub_name}.{axis}"
                    nodes[node_id] = Node(node_id, "axis", "l6", axis, params={"value": sub[axis], "sub_layer": sub_name})
    nodes["step:collect_inputs"] = Node(
        "step:collect_inputs",
        "step",
        "l6",
        "l6_collect_inputs",
        inputs=(
            NodeRef("src_l4_forecasts"),
            NodeRef("src_l4_model_artifacts"),
            NodeRef("src_l5_evaluation"),
            NodeRef("src_l1_data_definition"),
            NodeRef("src_l1_regime_metadata"),
            NodeRef("axis_enabled"),
        ),
    )
    previous = "step:collect_inputs"
    for sub_name in SUB_LAYER_ORDER:
        node_id = f"step:{sub_name}"
        axis_inputs = tuple(NodeRef(f"axis_{sub_name}.{axis}") for axis in SUB_LAYER_AXES[sub_name] if f"axis_{sub_name}.{axis}" in nodes)
        nodes[node_id] = Node(node_id, "step", "l6", sub_name, inputs=(NodeRef(previous),) + axis_inputs)
        previous = node_id
    nodes["sink:l6_tests_v1"] = Node("sink:l6_tests_v1", "sink", "l6", "sink", inputs=(NodeRef(previous),))
    return DAG("l6", nodes, sinks={"l6_tests_v1": "sink:l6_tests_v1"})


def resolve_axes(dag_or_layer: DAG | L6Layer | dict[str, Any]) -> L6ResolvedAxes:
    if isinstance(dag_or_layer, DAG):
        return getattr(dag_or_layer, "_l6_resolved", _resolve_from_dag_nodes(dag_or_layer))
    raw = dag_or_layer.raw_yaml if isinstance(dag_or_layer, L6Layer) else dag_or_layer
    return resolve_axes_from_raw(raw)


def resolve_axes_from_raw(raw: dict[str, Any], context: dict[str, Any] | None = None) -> L6ResolvedAxes:
    context = context or {}
    values: dict[str, Any] = {key: _copy_default(value) for key, value in GLOBAL_DEFAULTS.items()}
    for key in L6StatisticalTests.layer_globals:
        if key in raw:
            values[key] = raw[key]
    if "enabled" in raw:
        values["enabled"] = raw["enabled"]
    sub_layers_raw = raw.get("sub_layers", {}) or {}
    for sub_name in SUB_LAYER_ORDER:
        sub_raw = sub_layers_raw.get(sub_name, {}) or {}
        sub_values = {axis: _copy_default(value) for axis, value in SUB_LAYER_DEFAULTS[sub_name].items()}
        sub_values["enabled"] = bool(sub_raw.get("enabled", False)) and bool(values["enabled"])
        sub_values.update(sub_raw.get("fixed_axes", {}) or {})
        sub_active = {axis: bool(sub_values["enabled"]) for axis in SUB_LAYER_AXES[sub_name]}
        _apply_sub_layer_gates(sub_name, sub_values, sub_active, context)
        values[sub_name] = AxisGroup(sub_values, sub_active)
    active = {key: True for key in values if key not in SUB_LAYER_ORDER}
    resolved = L6ResolvedAxes(values, active)
    dag_marker = context.get("_dag_marker")
    if dag_marker is not None:
        setattr(dag_marker, "_l6_resolved", resolved)
    return resolved


def validate_layer(layer: dict[str, Any] | str, context: dict[str, Any] | None = None):
    from ..validator import ValidationReport

    raw = parse_layer_yaml(layer) if isinstance(layer, str) else layer
    context = context or {}
    resolved = resolve_axes_from_raw(raw, context=context)
    issues: list[Any] = []
    if _is_sweep(raw.get("enabled")):
        issues.append(_issue("l6.enabled", "L6 axes are not sweepable"))
    for axis in L6StatisticalTests.layer_globals:
        if _is_sweep(raw.get(axis)):
            issues.append(_issue(f"l6.{axis}", "L6 axes are not sweepable"))
    sub_layers_raw = raw.get("sub_layers", {}) or {}
    for sub_name, sub_raw_any in sub_layers_raw.items():
        if sub_name not in SUB_LAYER_AXES:
            issues.append(_issue(f"l6.{sub_name}", f"unknown L6 sub-layer {sub_name!r}"))
            continue
        sub_raw = sub_raw_any or {}
        fixed = sub_raw.get("fixed_axes", {}) or {}
        sub = resolved[sub_name]
        if bool(sub_raw.get("enabled", False)) and not resolved["enabled"]:
            continue
        if sub_name == "L6_E_density_interval" and bool(sub_raw.get("enabled", False)) and context.get("forecast_object", "point") not in {"quantile", "density"}:
            issues.append(_issue("l6.L6_E_density_interval", "L6.E requires quantile or density forecasts"))
        for axis, value in fixed.items():
            if axis not in SUB_LAYER_AXES[sub_name]:
                issues.append(_issue(f"l6.{sub_name}.{axis}", f"unknown L6 axis {axis!r}"))
            if _is_sweep(value):
                issues.append(_issue(f"l6.{sub_name}.{axis}", "L6 axes are not sweepable"))
            if axis in SUB_LAYER_AXES[sub_name] and not sub.get_active(axis):
                continue
        issues.extend(_validate_sub_layer(sub_name, sub, raw.get("leaf_config", {}) or {}, context))
    issues.extend(_validate_globals(resolved, context))
    return ValidationReport(tuple(issues))


def validate_recipe(recipe: L6Recipe | dict[str, Any] | str):
    from ..validator import ValidationReport

    obj = parse_recipe_yaml(recipe) if isinstance(recipe, (str, dict)) else recipe
    root = obj.root
    if "6_statistical_tests" not in root:
        return ValidationReport()
    return validate_layer(root["6_statistical_tests"] or {}, context=_recipe_context(root))


def make_recipe_without_benchmark() -> dict[str, Any]:
    return {
        "4_forecasting_model": {
            "forecast_object": "point",
            "nodes": [{"id": "fit_a", "type": "step", "op": "fit_model", "params": {"family": "ridge"}, "inputs": ["src_X", "src_y"]}],
        },
        "5_evaluation": {"fixed_axes": {}},
    }


def make_recipe_with_density_forecast() -> dict[str, Any]:
    root = make_recipe_without_benchmark()
    root["4_forecasting_model"]["forecast_object"] = "density"
    return root


def _recipe_context(root: dict[str, Any]) -> dict[str, Any]:
    l1_fixed = ((root.get("1_data", {}) or {}).get("fixed_axes", {}) or {})
    l1_leaf = ((root.get("1_data", {}) or {}).get("leaf_config", {}) or {})
    l4 = root.get("4_forecasting_model", {}) or {}
    l4_nodes = l4.get("nodes", ()) or ()
    benchmark_count = sum(1 for node in l4_nodes if isinstance(node, dict) and node.get("is_benchmark"))
    horizons = _extract_horizons(l1_fixed, l1_leaf)
    return {
        "benchmark_count": benchmark_count,
        "model_ids": _extract_model_ids(l4_nodes),
        "forecast_object": l4.get("forecast_object", "point"),
        "regime_definition": l1_fixed.get("regime_definition", "none"),
        "frequency": l1_fixed.get("frequency", "monthly"),
        "horizons": horizons,
    }


def _apply_sub_layer_gates(sub_name: str, values: dict[str, Any], active: dict[str, bool], context: dict[str, Any]) -> None:
    if sub_name == "L6_D_multiple_model":
        test = values.get("multiple_model_test")
        active["mcs_t_statistic"] = bool(values["enabled"]) and test in {"mcs_hansen", "multi"}
        active["spa_studentization"] = bool(values["enabled"]) and test in {"spa_hansen", "multi"}
        active["stepm_alpha"] = bool(values["enabled"]) and test in {"step_m_romano_wolf", "multi"}
    if sub_name == "L6_G_residual" and values.get("residual_lag_count") == "derived":
        values["residual_lag_count"] = 4 if context.get("frequency") == "quarterly" else 10


def _validate_globals(resolved: L6ResolvedAxes, context: dict[str, Any]) -> list[Any]:
    issues = []
    if resolved["overlap_handling"] == "none" and any(h > 1 for h in context.get("horizons", (1,))):
        issues.append(_issue("l6.overlap_handling", "overlap_handling=none is invalid when any horizon is greater than 1"))
    return issues


def _validate_sub_layer(sub_name: str, sub: AxisGroup, leaf: dict[str, Any], context: dict[str, Any]) -> list[Any]:
    issues = []
    if not sub["enabled"]:
        return issues
    if sub_name == "L6_A_equal_predictive":
        if sub["model_pair_strategy"] == "vs_benchmark_only" and context.get("benchmark_count", 0) != 1:
            issues.append(_issue("l6.L6_A_equal_predictive.model_pair_strategy", "vs_benchmark_only requires exactly one L4 benchmark model"))
        if sub["model_pair_strategy"] == "user_list":
            pairs = leaf.get("pair_user_list", [])
            model_ids = set(context.get("model_ids", ()))
            if not pairs or any(a not in model_ids or b not in model_ids for a, b in pairs):
                issues.append(_issue("l6.L6_A_equal_predictive.pair_user_list", "pair_user_list includes unknown model_id"))
    if sub_name == "L6_B_nested" and sub["nested_test"] == "clark_west" and sub["cw_adjustment"] is False:
        issues.append(_issue("l6.L6_B_nested.cw_adjustment", "Clark-West requires cw_adjustment=true"))
    if sub_name == "L6_C_cpa":
        if sub["cpa_conditioning_info"] == "regime" and context.get("regime_definition", "none") == "none":
            issues.append(_issue("l6.L6_C_cpa.cpa_conditioning_info", "regime conditioning requires active L1 regime"))
        if sub["cpa_conditioning_info"] == "external_indicator" and "cpa_external_indicator_path" not in leaf:
            issues.append(_issue("l6.L6_C_cpa.cpa_external_indicator_path", "external_indicator requires cpa_external_indicator_path"))
    if sub_name == "L6_D_multiple_model":
        if not (0 < float(sub["mcs_alpha"]) < 0.5):
            issues.append(_issue("l6.L6_D_multiple_model.mcs_alpha", "mcs_alpha must be in (0, 0.5)"))
        if int(sub["bootstrap_n_replications"]) < 100:
            issues.append(_issue("l6.L6_D_multiple_model.bootstrap_n_replications", "bootstrap_n_replications must be >= 100"))
        if not (0 < float(sub["stepm_alpha"]) < 0.5):
            issues.append(_issue("l6.L6_D_multiple_model.stepm_alpha", "stepm_alpha must be in (0, 0.5)"))
    if sub_name == "L6_E_density_interval":
        levels = sub["coverage_levels"]
        if not levels or any((not isinstance(level, (int, float))) or level <= 0 or level >= 1 for level in levels) or len(set(levels)) != len(levels):
            issues.append(_issue("l6.L6_E_density_interval.coverage_levels", "coverage_levels must be unique values in (0, 1)"))
        if int(sub["pit_n_bins"]) < 5:
            issues.append(_issue("l6.L6_E_density_interval.pit_n_bins", "pit_n_bins must be >= 5"))
        if sub["pit_test_horizon_dependence"] == "none" and any(h > 1 for h in context.get("horizons", (1,))):
            issues.append(_issue("l6.L6_E_density_interval.pit_test_horizon_dependence", "PIT horizon dependence none is invalid when h > 1"))
    if sub_name == "L6_F_direction":
        if sub["direction_threshold"] == "user_defined" and "direction_threshold_value" not in leaf:
            issues.append(_issue("l6.L6_F_direction.direction_threshold_value", "user_defined threshold requires direction_threshold_value"))
        if not (0 < float(sub["direction_alpha"]) < 0.5):
            issues.append(_issue("l6.L6_F_direction.direction_alpha", "direction_alpha must be in (0, 0.5)"))
    if sub_name == "L6_G_residual":
        if int(sub["residual_lag_count"]) < 1:
            issues.append(_issue("l6.L6_G_residual.residual_lag_count", "residual_lag_count must be >= 1"))
        if not (0 < float(sub["residual_alpha"]) < 0.5):
            issues.append(_issue("l6.L6_G_residual.residual_alpha", "residual_alpha must be in (0, 0.5)"))
    return issues


def _resolve_from_dag_nodes(dag: DAG) -> L6ResolvedAxes:
    values = {key: _copy_default(value) for key, value in GLOBAL_DEFAULTS.items()}
    for node in dag.nodes.values():
        if node.id.startswith("axis_") and "." not in node.id:
            values[node.op] = node.params["value"]
    for sub_name in SUB_LAYER_ORDER:
        sub_values = {axis: _copy_default(value) for axis, value in SUB_LAYER_DEFAULTS[sub_name].items()}
        sub_values["enabled"] = any(node.id.startswith(f"axis_{sub_name}.") for node in dag.nodes.values())
        sub_active = {axis: f"axis_{sub_name}.{axis}" in dag.nodes for axis in SUB_LAYER_AXES[sub_name]}
        for axis in SUB_LAYER_AXES[sub_name]:
            node = dag.nodes.get(f"axis_{sub_name}.{axis}")
            if node is not None:
                sub_values[axis] = node.params["value"]
        values[sub_name] = AxisGroup(sub_values, sub_active)
    return L6ResolvedAxes(values, {key: True for key in values if key not in SUB_LAYER_ORDER})


def _extract_horizons(l1_fixed: dict[str, Any], l1_leaf: dict[str, Any]) -> tuple[int, ...]:
    if l1_fixed.get("horizon_set") in {"custom_list", "single"}:
        return tuple(l1_leaf.get("target_horizons", (1,)))
    if l1_fixed.get("horizon_set") == "range_up_to_h":
        return tuple(range(1, int(l1_leaf.get("max_horizon", 1)) + 1))
    if l1_fixed.get("frequency") == "quarterly":
        return (1, 2, 4, 8)
    return (1, 3, 6, 12)


def _extract_model_ids(nodes: Any) -> tuple[str, ...]:
    return tuple(node.get("id") for node in nodes if isinstance(node, dict) and node.get("type") == "step" and node.get("op") == "fit_model" and node.get("id"))


def _copy_default(value: Any) -> Any:
    return list(value) if isinstance(value, list) else value


def _is_sweep(value: Any) -> bool:
    return isinstance(value, dict) and "sweep" in value


def _issue(location: str, message: str):
    from ..validator import Issue, Severity

    return Issue("l6_contract", Severity.HARD, "layer", location, message)
