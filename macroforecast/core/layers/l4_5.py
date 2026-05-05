from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..dag import DAG, Node, NodeRef, SourceSelector


@dataclass(frozen=True)
class SubLayerSpec:
    axes: tuple[str, ...]


class L4_5GeneratorDiagnostics:
    """Layer 4.5 Generator Diagnostics implementation marker."""

    sub_layers = {
        "L4_5_A_in_sample_fit": SubLayerSpec(("fit_view", "fit_per_origin")),
        "L4_5_B_forecast_scale_view": SubLayerSpec(("forecast_scale_view", "back_transform_method")),
        "L4_5_C_window_stability": SubLayerSpec(("window_view", "coef_view_models")),
        "L4_5_D_tuning_history": SubLayerSpec(("tuning_view",)),
        "L4_5_E_ensemble_diagnostics": SubLayerSpec(("ensemble_view", "weights_over_time_method")),
        "L4_5_Z_export": SubLayerSpec(("diagnostic_format", "attach_to_manifest", "figure_dpi", "latex_export")),
    }

    @classmethod
    def list_axes(cls) -> tuple[str, ...]:
        return tuple(axis for spec in cls.sub_layers.values() for axis in spec.axes)

    @classmethod
    def list_sublayers(cls) -> tuple[str, ...]:
        return tuple(cls.sub_layers)


class L4_5ResolvedAxes(dict):
    def __init__(self, values: dict[str, Any], active: dict[str, bool]) -> None:
        super().__init__(values)
        self._active = active

    def get_active(self, key: str) -> bool:
        return self._active.get(key, True)


AXIS_NAMES = L4_5GeneratorDiagnostics.list_axes()

DEFAULT_AXES: dict[str, Any] = {
    "fit_view": "multi",
    "fit_per_origin": "last_origin_only",
    "forecast_scale_view": "both_overlay",
    "back_transform_method": "auto",
    "window_view": "multi",
    "coef_view_models": "all_linear_models",
    "tuning_view": "multi",
    "ensemble_view": "multi",
    "weights_over_time_method": "stacked_area",
    "diagnostic_format": "pdf",
    "attach_to_manifest": True,
    "figure_dpi": 300,
    "latex_export": True,
}

OPTIONS = {
    "fit_view": {"fitted_vs_actual", "residual_time", "residual_acf", "residual_qq", "multi"},
    "fit_per_origin": {"last_origin_only", "every_n_origins", "all_origins"},
    "forecast_scale_view": {"transformed_only", "back_transformed_only", "both_overlay"},
    "back_transform_method": {"auto", "manual_function"},
    "window_view": {"rolling_train_loss", "rolling_coef", "first_vs_last_window_forecast", "parameter_stability", "multi"},
    "coef_view_models": {"all_linear_models", "user_list"},
    "tuning_view": {"objective_trace", "hyperparameter_path", "cv_score_distribution", "multi"},
    "ensemble_view": {"weights_over_time", "weight_concentration", "member_contribution", "multi"},
    "weights_over_time_method": {"line_plot", "stacked_area", "heatmap"},
    "diagnostic_format": {"png", "pdf", "html", "json", "latex_table", "csv", "multi"},
}

LINEAR_FAMILIES = {
    "ols",
    "ridge",
    "lasso",
    "elastic_net",
    "var",
    "bvar_minnesota",
    "bvar_normal_inverse_wishart",
    "ar_p",
    "ar1",
    "ardl",
    "factor_augmented_ar",
}
ENSEMBLE_OPS = {"weighted_average_forecast", "median_forecast", "trimmed_mean_forecast", "bma_forecast", "bivariate_ardl_combination"}


def parse_layer_yaml(yaml_text: str, layer_id: Literal["l4_5"] = "l4_5") -> dict[str, Any]:
    if layer_id != "l4_5":
        raise ValueError("L4.5 parser only accepts layer_id='l4_5'")
    from ..yaml import parse_recipe_yaml

    root = parse_recipe_yaml(yaml_text)
    raw = root.get("4_5_generator_diagnostics", root)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("4_5_generator_diagnostics: layer YAML must be a mapping")
    return raw


def parse_recipe_yaml(yaml_text_or_root: str | dict[str, Any]) -> Any:
    from ..yaml import parse_recipe_yaml as parse

    root = parse(yaml_text_or_root) if isinstance(yaml_text_or_root, str) else yaml_text_or_root
    return L4_5Recipe(root)


@dataclass(frozen=True)
class L4_5Layer:
    raw_yaml: dict[str, Any]
    dag: DAG
    enabled: bool = False


@dataclass(frozen=True)
class L4_5Recipe:
    root: dict[str, Any]

    @property
    def layers(self) -> dict[str, Any]:
        if "4_5_generator_diagnostics" not in self.root:
            return {}
        raw = self.root["4_5_generator_diagnostics"] or {}
        enabled = bool(raw.get("enabled", False))
        return {"l4_5": L4_5Layer(raw, normalize_to_dag_form(raw, "l4_5", context=_recipe_context(self.root)), enabled)}


def normalize_to_dag_form(layer: dict[str, Any] | L4_5Layer, layer_id: Literal["l4_5"] = "l4_5", context: dict[str, Any] | None = None) -> DAG:
    raw = layer.raw_yaml if isinstance(layer, L4_5Layer) else layer
    resolved = resolve_axes_from_raw(raw, context=context)
    if not resolved["enabled"]:
        return DAG("l4_5", {}, sinks={}, layer_globals={"resolved_axes": resolved})
    nodes: dict[str, Node] = {
        "src_l4_forecasts": Node("src_l4_forecasts", "source", "l4_5", "source", selector=SourceSelector("l4", "l4_forecasts_v1")),
        "src_l4_models": Node("src_l4_models", "source", "l4_5", "source", selector=SourceSelector("l4", "l4_model_artifacts_v1")),
        "src_l4_training": Node("src_l4_training", "source", "l4_5", "source", selector=SourceSelector("l4", "l4_training_metadata_v1")),
        "src_l3_features": Node("src_l3_features", "source", "l4_5", "source", selector=SourceSelector("l3", "l3_features_v1")),
    }
    for axis in AXIS_NAMES:
        if resolved.get_active(axis):
            nodes[f"axis_{axis}"] = Node(f"axis_{axis}", "axis", "l4_5", axis, params={"value": resolved[axis]})
    previous = "step:collect_l4"
    nodes[previous] = Node(
        previous,
        "step",
        "l4_5",
        "diagnostic_collect_l4",
        inputs=(NodeRef("src_l4_forecasts"), NodeRef("src_l4_models"), NodeRef("src_l4_training"), NodeRef("src_l3_features")),
    )
    for step in ("in_sample_fit", "forecast_scale_view", "window_stability", "tuning_history", "ensemble_diagnostics", "diagnostic_export"):
        node_id = f"step:{step}"
        axis_inputs = tuple(NodeRef(f"axis_{axis}") for axis in _step_axes(step) if f"axis_{axis}" in nodes)
        nodes[node_id] = Node(node_id, "step", "l4_5", f"l4_5_{step}", inputs=(NodeRef(previous),) + axis_inputs)
        previous = node_id
    nodes["sink:l4_5_diagnostic_v1"] = Node("sink:l4_5_diagnostic_v1", "sink", "l4_5", "sink", inputs=(NodeRef(previous),))
    return DAG("l4_5", nodes, sinks={"l4_5_diagnostic_v1": "sink:l4_5_diagnostic_v1"}, layer_globals={"resolved_axes": resolved})


def resolve_axes(dag_or_layer: DAG | L4_5Layer | dict[str, Any]) -> L4_5ResolvedAxes:
    if isinstance(dag_or_layer, DAG):
        return dag_or_layer.layer_globals.get("resolved_axes", resolve_axes_from_raw({}))
    raw = dag_or_layer.raw_yaml if isinstance(dag_or_layer, L4_5Layer) else dag_or_layer
    return resolve_axes_from_raw(raw)


def resolve_axes_from_raw(raw: dict[str, Any], context: dict[str, Any] | None = None) -> L4_5ResolvedAxes:
    fixed = raw.get("fixed_axes", {}) or {}
    leaf = raw.get("leaf_config", {}) or {}
    values = {axis: _copy_default(default) for axis, default in DEFAULT_AXES.items()}
    values.update(fixed)
    values["enabled"] = bool(raw.get("enabled", False))
    values["leaf_config"] = {
        "fit_n_origins_step": leaf.get("fit_n_origins_step", 12),
        "coef_top_k": leaf.get("coef_top_k", 10),
        **leaf,
    }
    active = {axis: True for axis in AXIS_NAMES}
    context = context or {}
    if not values["enabled"]:
        active = {axis: False for axis in AXIS_NAMES}
    if not context.get("has_linear_model", False):
        active["coef_view_models"] = False
    if not context.get("has_tuning", False):
        active["tuning_view"] = False
    if not context.get("has_ensemble", False):
        active["ensemble_view"] = False
        active["weights_over_time_method"] = False
    if values["ensemble_view"] not in {"weights_over_time", "multi"}:
        active["weights_over_time_method"] = False
    return L4_5ResolvedAxes(values, active)


def validate_layer(layer: dict[str, Any] | str, context: dict[str, Any] | None = None):
    from ..validator import ValidationReport

    raw = parse_layer_yaml(layer) if isinstance(layer, str) else layer
    context = context or {}
    fixed = raw.get("fixed_axes", {}) or {}
    leaf = raw.get("leaf_config", {}) or {}
    resolved = resolve_axes_from_raw(raw, context=context)
    issues: list[Any] = []
    for axis, value in fixed.items():
        if axis not in AXIS_NAMES:
            issues.append(_issue(f"l4_5.{axis}", f"unknown L4.5 axis {axis!r}"))
        if _is_sweep(value):
            issues.append(_issue(f"l4_5.{axis}", "Diagnostic axes are not sweepable"))
    issues.extend(_validate_values(resolved, fixed, leaf, context))
    return ValidationReport(tuple(issues))


def validate_recipe(recipe: L4_5Recipe | dict[str, Any] | str):
    from ..validator import ValidationReport

    obj = parse_recipe_yaml(recipe) if isinstance(recipe, (str, dict)) else recipe
    if "4_5_generator_diagnostics" not in obj.root:
        return ValidationReport()
    return validate_layer(obj.root["4_5_generator_diagnostics"] or {}, context=_recipe_context(obj.root))


def _validate_values(resolved: L4_5ResolvedAxes, fixed: dict[str, Any], leaf: dict[str, Any], context: dict[str, Any]) -> list[Any]:
    issues: list[Any] = []
    if not resolved["enabled"]:
        return issues
    for axis, options in OPTIONS.items():
        if resolved.get_active(axis) and isinstance(resolved[axis], str) and resolved[axis] not in options:
            issues.append(_issue(f"l4_5.{axis}", f"invalid value {resolved[axis]!r} for {axis}"))
    if resolved["back_transform_method"] == "manual_function" and "back_transform_function" not in leaf:
        issues.append(_issue("l4_5.back_transform_method", "manual_function requires leaf_config.back_transform_function"))
    if "coef_view_models" in fixed and not context.get("has_linear_model", False):
        issues.append(_issue("l4_5.coef_view_models", "coef_view_models requires at least one linear model in L4"))
    if "tuning_view" in fixed and not context.get("has_tuning", False):
        issues.append(_issue("l4_5.tuning_view", "tuning_view requires at least one L4 model with search_algorithm != none"))
    if "ensemble_view" in fixed and not context.get("has_ensemble", False):
        issues.append(_issue("l4_5.ensemble_view", "ensemble_view requires an L4 ensemble combine node"))
    return issues


def _recipe_context(root: dict[str, Any]) -> dict[str, Any]:
    nodes = ((root.get("4_forecasting_model") or {}).get("nodes") or [])
    families: set[str] = set()
    search_algorithms: set[str] = set()
    ops: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict):
            continue
        op = node.get("op")
        if op:
            ops.add(str(op))
        params = node.get("params") or {}
        if isinstance(params, dict):
            if "family" in params:
                families.add(str(params["family"]))
            if "search_algorithm" in params:
                search_algorithms.add(str(params["search_algorithm"]))
    return {
        "has_linear_model": bool(families & LINEAR_FAMILIES),
        "has_tuning": any(algorithm != "none" for algorithm in search_algorithms),
        "has_ensemble": bool(ops & ENSEMBLE_OPS),
    }


def _step_axes(step: str) -> tuple[str, ...]:
    return {
        "in_sample_fit": ("fit_view", "fit_per_origin"),
        "forecast_scale_view": ("forecast_scale_view", "back_transform_method"),
        "window_stability": ("window_view", "coef_view_models"),
        "tuning_history": ("tuning_view",),
        "ensemble_diagnostics": ("ensemble_view", "weights_over_time_method"),
        "diagnostic_export": ("diagnostic_format", "attach_to_manifest", "figure_dpi", "latex_export"),
    }[step]


def _is_sweep(value: Any) -> bool:
    return isinstance(value, dict) and "sweep" in value


def _copy_default(value: Any) -> Any:
    return list(value) if isinstance(value, list) else value


def _issue(path: str, message: str) -> Any:
    from ..validator import Issue, Severity

    return Issue("l4_5_contract", Severity.HARD, "layer", path, message)


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
        sweepable=False,
    )


_SUBLAYER_NAMES = {
    "L4_5_A_in_sample_fit": "In-sample fit",
    "L4_5_B_forecast_scale_view": "Forecast scale view",
    "L4_5_C_window_stability": "Window stability",
    "L4_5_D_tuning_history": "Tuning history",
    "L4_5_E_ensemble_diagnostics": "Ensemble diagnostics",
    "L4_5_Z_export": "Diagnostic export",
}


L4_5_LAYER_SPEC = _LayerImplSpec(
    layer_id="l4_5",
    name="Generator diagnostics",
    category="diagnostic",
    expected_inputs=("l4_forecasts_v1", "l4_model_artifacts_v1", "l4_training_metadata_v1"),
    produces=("l4_5_diagnostic_v1",),
    ui_mode="list",
    layer_globals=("enabled",),
    sub_layers=tuple(
        _CanonicalSubLayerSpec(id=sl_id, name=_SUBLAYER_NAMES[sl_id], axes=spec.axes)
        for sl_id, spec in L4_5GeneratorDiagnostics.sub_layers.items()
    ),
    axes={
        sl_id: {axis: _build_axis(axis) for axis in spec.axes}
        for sl_id, spec in L4_5GeneratorDiagnostics.sub_layers.items()
    },
)
