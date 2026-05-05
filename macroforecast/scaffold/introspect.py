"""Walk the :class:`LayerImplementationSpec` registry and surface
axis/option metadata in a form the wizard UI and docs site can consume.

The wizard never reads ``LayerImplementationSpec`` directly -- it goes
through this module so a future schema-shape change has a single
re-wire point.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

from ..core.layer_specs import AxisSpec, LayerImplementationSpec, Option, SubLayerSpec


# Layer modules that expose ``<LAYER>_LAYER_SPEC`` constants. Order
# matches the canonical layer-execution order so wizard prompts walk in
# dependency order.
_LAYER_MODULES: tuple[tuple[str, str], ...] = (
    ("l0", "macroforecast.core.layers.l0"),
    ("l1", "macroforecast.core.layers.l1"),
    ("l1_5", "macroforecast.core.layers.l1_5"),
    ("l2", "macroforecast.core.layers.l2"),
    ("l2_5", "macroforecast.core.layers.l2_5"),
    ("l3", "macroforecast.core.layers.l3"),
    ("l3_5", "macroforecast.core.layers.l3_5"),
    ("l4", "macroforecast.core.layers.l4"),
    ("l4_5", "macroforecast.core.layers.l4_5"),
    ("l5", "macroforecast.core.layers.l5"),
    ("l6", "macroforecast.core.layers.l6"),
    ("l7", "macroforecast.core.layers.l7"),
    ("l8", "macroforecast.core.layers.l8"),
)


@dataclass(frozen=True)
class OptionInfo:
    """Wizard-friendly view of one option on one axis."""

    value: str
    label: str
    description: str
    status: str = "operational"
    leaf_config_required: tuple[str, ...] = ()
    leaf_config_optional: tuple[str, ...] = ()


@dataclass(frozen=True)
class AxisInfo:
    """Wizard-friendly view of one axis on one sub-layer."""

    layer: str
    sublayer: str
    name: str
    default: Any
    status: str
    sweepable: bool
    options: tuple[OptionInfo, ...]
    has_gate: bool
    leaf_config_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class SubLayerInfo:
    """Wizard-friendly view of one sub-layer on one layer."""

    id: str
    name: str
    has_gate: bool
    axis_names: tuple[str, ...]


@dataclass(frozen=True)
class LayerInfo:
    """Wizard-friendly view of one layer."""

    id: str
    name: str
    category: str
    sub_layers: tuple[SubLayerInfo, ...]
    layer_globals: tuple[str, ...]


def _load_spec(layer_id: str) -> LayerImplementationSpec | None:
    """Import the layer's module and return its ``<LAYER>_LAYER_SPEC``
    constant; ``None`` when the module exists but doesn't expose one."""

    module_path = dict(_LAYER_MODULES).get(layer_id)
    if module_path is None:
        return None
    module = import_module(module_path)
    spec_attr = f"{layer_id.upper()}_LAYER_SPEC"
    return getattr(module, spec_attr, None)


def list_layers() -> tuple[str, ...]:
    """Return the ordered tuple of layer identifiers."""

    return tuple(layer_id for layer_id, _ in _LAYER_MODULES)


def layer(layer_id: str) -> LayerInfo:
    """Return a ``LayerInfo`` summary for the supplied layer."""

    spec = _load_spec(layer_id)
    if spec is None:
        raise KeyError(f"unknown layer {layer_id!r}")
    sub_layers = tuple(
        SubLayerInfo(
            id=sub.id,
            name=sub.name,
            has_gate=sub.gate is not None,
            axis_names=tuple(sub.axes),
        )
        for sub in spec.sub_layers
    )
    return LayerInfo(
        id=spec.layer_id,
        name=spec.name,
        category=str(spec.category),
        sub_layers=sub_layers,
        layer_globals=tuple(spec.layer_globals),
    )


def axes(layer_id: str) -> tuple[AxisInfo, ...]:
    """Return every axis of the supplied layer flattened across sub-layers,
    in declaration order. When the ``LayerImplementationSpec`` is sparse
    (L3 / L4 / L6 declare axes via module constants rather than via the
    spec), fall back to the manual registry below."""

    spec = _load_spec(layer_id)
    if spec is None:
        return ()
    out: list[AxisInfo] = []
    seen: set[str] = set()
    fallback_by_name: dict[str, AxisInfo] = {
        a.name: a for a in _MANUAL_AXES.get(layer_id, ())
    }
    for sub in spec.sub_layers:
        sub_axes = spec.axes.get(sub.id, {})
        for axis_name in sub.axes:
            if axis_name in seen:
                continue
            seen.add(axis_name)
            axis_spec = sub_axes.get(axis_name)
            fallback = fallback_by_name.get(axis_name)
            if axis_spec is None and fallback is not None:
                out.append(fallback)
                continue
            if axis_spec is None:
                out.append(
                    AxisInfo(
                        layer=layer_id,
                        sublayer=sub.id,
                        name=axis_name,
                        default=None,
                        status="unknown",
                        sweepable=False,
                        options=(),
                        has_gate=False,
                    )
                )
                continue
            info = _axis_info(layer_id, sub.id, axis_spec)
            # If the spec axis has no enumerated options but a manual
            # fallback does, prefer the fallback (keeps the spec metadata
            # shape but unlocks the option list).
            if not info.options and fallback is not None:
                info = AxisInfo(
                    layer=info.layer, sublayer=info.sublayer, name=info.name,
                    default=info.default if info.default is not None else fallback.default,
                    status=info.status, sweepable=info.sweepable,
                    options=fallback.options, has_gate=info.has_gate,
                    leaf_config_keys=info.leaf_config_keys,
                )
            out.append(info)
    # Manual-only axes (declared in fallback but absent from spec).
    for entry in _MANUAL_AXES.get(layer_id, ()):
        if entry.name in seen:
            continue
        seen.add(entry.name)
        out.append(entry)
    return tuple(out)


def _axis_info(layer_id: str, sublayer_id: str, axis: AxisSpec) -> AxisInfo:
    options = tuple(_option_info(option) for option in axis.options)
    return AxisInfo(
        layer=layer_id,
        sublayer=sublayer_id,
        name=axis.name,
        default=axis.default,
        status=str(axis.status),
        sweepable=bool(axis.sweepable),
        options=options,
        has_gate=axis.gate is not None,
        leaf_config_keys=tuple(axis.leaf_config_keys),
    )


def _option_info(option: Option) -> OptionInfo:
    return OptionInfo(
        value=str(option.value),
        label=str(option.label),
        description=str(option.description),
        status=str(option.status),
        leaf_config_required=tuple(option.leaf_config_required),
        leaf_config_optional=tuple(option.leaf_config_optional),
    )


def _build_l4_fallback() -> tuple[AxisInfo, ...]:
    """L4's spec lists sub-layers but stores axes via module constants.
    Surface ``family`` / ``forecast_strategy`` / ``training_start_rule``
    / ``refit_policy`` / ``search_algorithm`` here so the wizard + sphinx
    docs can iterate them."""

    from ..core.ops.l4_ops import OPERATIONAL_MODEL_FAMILIES

    family_options = tuple(
        OptionInfo(value=fam, label=fam.replace("_", " ").title(), description="")
        for fam in OPERATIONAL_MODEL_FAMILIES
    )
    return (
        AxisInfo(
            layer="l4",
            sublayer="L4_A_model_selection",
            name="family",
            default="ridge",
            status="operational",
            sweepable=True,
            options=family_options,
            has_gate=False,
        ),
        AxisInfo(
            layer="l4",
            sublayer="L4_B_forecast_strategy",
            name="forecast_strategy",
            default="direct",
            status="operational",
            sweepable=True,
            options=(
                OptionInfo("direct", "Direct h-step", "One model per horizon."),
                OptionInfo("iterated", "Iterated", "h=1 model applied recursively."),
                OptionInfo("path_average", "Path average", "Cumulative-average target."),
            ),
            has_gate=False,
        ),
        AxisInfo(
            layer="l4",
            sublayer="L4_C_training_window",
            name="training_start_rule",
            default="expanding",
            status="operational",
            sweepable=True,
            options=(
                OptionInfo("expanding", "Expanding window", "Start at sample t=0."),
                OptionInfo("rolling", "Rolling window", "Fixed-size window of size rolling_window."),
                OptionInfo("fixed", "Fixed window", "Fixed start/end dates from leaf_config."),
            ),
            has_gate=False,
        ),
        AxisInfo(
            layer="l4",
            sublayer="L4_C_training_window",
            name="refit_policy",
            default="every_origin",
            status="operational",
            sweepable=True,
            options=(
                OptionInfo("every_origin", "Refit each origin", "Walk-forward default."),
                OptionInfo("every_n_origins", "Refit every n origins", "Caps refit cost."),
                OptionInfo("single_fit", "Single fit", "Fit once on the full sample."),
            ),
            has_gate=False,
        ),
        AxisInfo(
            layer="l4",
            sublayer="L4_D_tuning",
            name="search_algorithm",
            default="none",
            status="operational",
            sweepable=True,
            options=(
                OptionInfo("none", "No tuning", "Use the params as-is."),
                OptionInfo("cv_path", "Regularisation path", "RidgeCV / LassoCV path."),
                OptionInfo("grid_search", "Grid search", "Exhaustive over leaf_config.tuning_grid."),
                OptionInfo("random_search", "Random search", "Sample tuning_distributions."),
                OptionInfo("bayesian_optimization", "Bayesian (optuna)", "TPE optimisation."),
                OptionInfo("genetic_algorithm", "Genetic", "Tournament evolution."),
            ),
            has_gate=False,
        ),
    )


def _build_l3_fallback() -> tuple[AxisInfo, ...]:
    """L3 declares ops via the universal/l3 ops registry rather than as
    AxisSpec options. We surface a single ``op`` axis so docs + wizard
    can iterate the available L3 step ops."""

    try:
        from ..core.ops.registry import _OPS as _OPS_REGISTRY
    except ImportError:
        return ()
    options = tuple(
        sorted(
            (
                OptionInfo(value=name, label=name, description=getattr(spec, "description", "") or "")
                for name, spec in _OPS_REGISTRY.items()
                if "l3" in (spec.layer_scope or ()) and spec.status == "operational"
            ),
            key=lambda o: o.value,
        )
    )
    return (
        AxisInfo(
            layer="l3",
            sublayer="L3_A_step_op",
            name="op",
            default="lag",
            status="operational",
            sweepable=False,
            options=options,
            has_gate=False,
        ),
    )


def _build_l6_fallback() -> tuple[AxisInfo, ...]:
    """L6 stores test selectors as module constants. Surface the per-sub-layer
    test axis so docs + wizard can iterate them."""

    return (
        AxisInfo(
            layer="l6", sublayer="L6_A_equal_predictive", name="equal_predictive_test",
            default="dm_diebold_mariano", status="operational", sweepable=True,
            options=(
                OptionInfo("dm_diebold_mariano", "Diebold-Mariano", "DM with HLN small-sample correction."),
                OptionInfo("gw_giacomini_white", "Giacomini-White", "Conditional predictive ability."),
                OptionInfo("dmp_multi_horizon", "DM-Pesaran joint", "Joint multi-horizon test."),
                OptionInfo("multi", "All of the above", "Run every test in this sub-layer."),
            ),
            has_gate=False,
        ),
        AxisInfo(
            layer="l6", sublayer="L6_C_cpa", name="cpa_test",
            default="giacomini_rossi_2010", status="operational", sweepable=True,
            options=(
                OptionInfo("giacomini_rossi_2010", "Giacomini-Rossi rolling", "Rolling-window fluctuation test."),
                OptionInfo("rossi_sekhposyan", "Rossi-Sekhposyan", "Recursive variant."),
                OptionInfo("multi", "Both", "Run both variants."),
            ),
            has_gate=False,
        ),
        AxisInfo(
            layer="l6", sublayer="L6_D_multiple_model", name="multiple_model_test",
            default="mcs_hansen", status="operational", sweepable=True,
            options=(
                OptionInfo("mcs_hansen", "Hansen MCS", "Stationary block bootstrap."),
                OptionInfo("spa_hansen", "Hansen SPA", "Studentized SPA."),
                OptionInfo("reality_check_white", "White reality check", "Original RC."),
                OptionInfo("step_m_romano_wolf", "Romano-Wolf StepM", "Step-down."),
            ),
            has_gate=False,
        ),
    )


def _build_l7_fallback() -> tuple[AxisInfo, ...]:
    """L7.A is a DAG body whose 30 operational importance ops live in the
    universal op registry rather than as schema AxisSpec options. Surface
    a single ``op`` axis on ``L7_A_importance_dag_body`` so docs + wizard
    iterate every importance op consistently with L3/L4/L6 layers.

    Architectural parity: L3 (37 ops via universal registry), L4 (35
    families via OPERATIONAL_MODEL_FAMILIES), L6 (test selectors via
    module constants), L7 (30 ops via universal registry) all use this
    same fallback pattern.
    """

    try:
        from ..core.ops.registry import _OPS as _OPS_REGISTRY
    except ImportError:
        return ()
    options = tuple(
        sorted(
            (
                OptionInfo(
                    value=name,
                    label=name,
                    description=getattr(spec, "description", "") or "",
                )
                for name, spec in _OPS_REGISTRY.items()
                if "l7" in (spec.layer_scope or ()) and spec.status == "operational"
            ),
            key=lambda o: o.value,
        )
    )
    return (
        AxisInfo(
            layer="l7",
            sublayer="L7_A_importance_dag_body",
            name="op",
            default="permutation_importance",
            status="operational",
            sweepable=False,
            options=options,
            has_gate=False,
        ),
    )


_MANUAL_AXES: dict[str, tuple[AxisInfo, ...]] = {}


def _populate_manual_axes() -> None:
    global _MANUAL_AXES
    _MANUAL_AXES = {
        "l3": _build_l3_fallback(),
        "l4": _build_l4_fallback(),
        "l6": _build_l6_fallback(),
        "l7": _build_l7_fallback(),
    }


_populate_manual_axes()


def operational_options(layer_id: str) -> tuple[tuple[str, str, str, str], ...]:
    """Return ``(layer, sublayer, axis, option)`` tuples for every
    operational option on every operational axis. Used by the
    completeness test to enforce documentation coverage."""

    out: list[tuple[str, str, str, str]] = []
    for axis_info in axes(layer_id):
        if axis_info.status != "operational":
            continue
        for option in axis_info.options:
            if option.status != "operational":
                continue
            out.append((layer_id, axis_info.sublayer, axis_info.name, option.value))
    return tuple(out)


__all__ = [
    "AxisInfo",
    "LayerInfo",
    "OptionInfo",
    "SubLayerInfo",
    "axes",
    "layer",
    "list_layers",
    "operational_options",
]
