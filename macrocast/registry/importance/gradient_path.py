from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name="importance_gradient_path",
    layer="7_importance",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="none", description="no gradient-path summary", status="operational", priority="A"),
        EnumRegistryEntry(id="coefficient_path", description="coefficient / contribution path summary", status="planned", priority="B"),
    ),
    compatible_with={},
    incompatible_with={},
)
