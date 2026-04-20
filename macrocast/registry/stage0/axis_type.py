from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_TYPE_ENTRIES: tuple[EnumRegistryEntry, ...] = (
    EnumRegistryEntry(id="fixed", description="Axis is normally fixed within one study path.", status="operational", priority="A"),
    EnumRegistryEntry(id="sweep", description="Axis is normally swept across multiple values.", status="operational", priority="A"),
    EnumRegistryEntry(id="nested_sweep", description="Axis participates in nested sweep designs.", status="operational", priority="A"),
    EnumRegistryEntry(id="conditional", description="Axis is activated conditionally on other choices.", status="operational", priority="A"),
    EnumRegistryEntry(id="derived", description="Axis is derived from other recipe state.", status="operational", priority="A"),
)

AXIS_DEFINITION = AxisDefinition(
    axis_name="axis_type",
    layer="0_meta",
    axis_type="enum",
    default_policy="fixed",
    entries=AXIS_TYPE_ENTRIES,
    compatible_with={},
    incompatible_with={},
)
