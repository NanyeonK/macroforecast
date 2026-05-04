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
    ("l0", "macrocast.core.layers.l0"),
    ("l1", "macrocast.core.layers.l1"),
    ("l1_5", "macrocast.core.layers.l1_5"),
    ("l2", "macrocast.core.layers.l2"),
    ("l2_5", "macrocast.core.layers.l2_5"),
    ("l3", "macrocast.core.layers.l3"),
    ("l3_5", "macrocast.core.layers.l3_5"),
    ("l4", "macrocast.core.layers.l4"),
    ("l4_5", "macrocast.core.layers.l4_5"),
    ("l5", "macrocast.core.layers.l5"),
    ("l6", "macrocast.core.layers.l6"),
    ("l7", "macrocast.core.layers.l7"),
    ("l8", "macrocast.core.layers.l8"),
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
    in declaration order."""

    spec = _load_spec(layer_id)
    if spec is None:
        return ()
    out: list[AxisInfo] = []
    seen: set[str] = set()
    for sub in spec.sub_layers:
        sub_axes = spec.axes.get(sub.id, {})
        for axis_name in sub.axes:
            if axis_name in seen:
                # Same axis declared on multiple sub-layers -- emit once.
                continue
            seen.add(axis_name)
            axis_spec = sub_axes.get(axis_name)
            if axis_spec is None:
                # Axis declared on the sub-layer but no AxisSpec entry --
                # surface a placeholder so callers can still iterate the
                # gate-aware structure.
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
            out.append(_axis_info(layer_id, sub.id, axis_spec))
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
