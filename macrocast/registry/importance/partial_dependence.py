from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name="importance_partial_dependence",
    layer="7_importance",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="pdp", description="partial dependence", status="operational", priority="A"),
        EnumRegistryEntry(id="ice", description="individual conditional expectation", status="operational", priority="A"),
        EnumRegistryEntry(id="ale", description="accumulated local effects", status="operational", priority="A"),
    ),
    compatible_with={},
    incompatible_with={},
)
