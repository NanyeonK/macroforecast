from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name="importance_scope",
    layer="7_importance",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="global", description="global importance summary", status="operational", priority="A"),
        EnumRegistryEntry(id="local", description="single-prediction local explanation", status="operational", priority="A"),
        EnumRegistryEntry(id="both", description="global and local outputs", status="registry_only", priority="B"),
    ),
    compatible_with={},
    incompatible_with={},
)
