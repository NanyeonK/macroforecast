"""Form schema converter: AxisInfo -> FormField.

Converts the introspection layer schema into a flat list of FormField
objects that drive the Solara form components.

Usage:
    from macroforecast.wizard.schema import layer_form_schema, option_doc_for
    fields = layer_form_schema("l0")
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from macroforecast.scaffold import introspect
from macroforecast.scaffold.option_docs import OPTION_DOCS
from macroforecast.scaffold.option_docs.types import OptionDoc


@dataclass
class FormField:
    """Wizard-friendly view of one axis, ready to render as a form widget."""

    axis_name: str
    sublayer: str
    label: str
    """axis_name formatted for display (replace _ with space, title-case)."""
    widget_type: str
    """'select' | 'text' | 'int' | 'float' | 'bool'"""
    options: list[tuple[str, str]]
    """[(value, display_label), ...] for select; empty for others."""
    default: Any
    is_sweepable: bool
    leaf_config_keys: list[str]
    """Leaf keys that become visible when this axis is selected."""
    doc: OptionDoc | None
    """None when not registered in OPTION_DOCS."""
    status: str
    """'operational' | 'future'"""


def _widget_type(axis: introspect.AxisInfo) -> str:
    """Determine widget_type from AxisInfo."""
    if len(axis.options) > 0:
        return "select"
    if axis.default is True or axis.default is False:
        return "bool"
    if isinstance(axis.default, bool):
        return "bool"
    if isinstance(axis.default, int):
        return "int"
    if isinstance(axis.default, float):
        return "float"
    return "text"


def _make_label(axis_name: str) -> str:
    """Format axis_name for display: replace _ with space, title-case."""
    return axis_name.replace("_", " ").title()


def _make_options(axis: introspect.AxisInfo) -> list[tuple[str, str]]:
    """Build (value, display_label) pairs from AxisInfo.options."""
    result = []
    for opt in axis.options:
        label = opt.label or opt.description or opt.value
        result.append((opt.value, label))
    return result


def layer_form_schema(layer_id: str) -> list[FormField]:
    """Convert all operational axes for layer_id into a list of FormField.

    Args:
        layer_id: e.g. "l0", "l1", "l2", "l5", "l6"

    Returns:
        List of FormField in declaration order.

    Raises:
        KeyError: if layer_id is unknown.
    """
    try:
        all_axes = introspect.axes(layer_id)
    except KeyError:
        raise KeyError(f"wizard: unknown layer_id={layer_id!r}")

    fields: list[FormField] = []
    for axis in all_axes:
        if axis.status != "operational":
            continue

        wtype = _widget_type(axis)
        opts = _make_options(axis)
        label = _make_label(axis.name)

        # Axis-level doc: not per-option at this stage; OptionInput fetches
        # per-option docs when the user clicks "?".
        doc: OptionDoc | None = None

        fields.append(
            FormField(
                axis_name=axis.name,
                sublayer=axis.sublayer,
                label=label,
                widget_type=wtype,
                options=opts,
                default=axis.default,
                is_sweepable=axis.sweepable,
                leaf_config_keys=list(axis.leaf_config_keys),
                doc=doc,
                status=axis.status,
            )
        )

    return fields


def option_doc_for(
    layer_id: str,
    sublayer: str,
    axis_name: str,
    option_value: str,
) -> OptionDoc | None:
    """Look up an OptionDoc for a specific (layer, sublayer, axis, option) tuple.

    Returns None when no doc is registered yet.
    """
    return OPTION_DOCS.get((layer_id, sublayer, axis_name, option_value))


__all__ = ["FormField", "layer_form_schema", "option_doc_for"]
