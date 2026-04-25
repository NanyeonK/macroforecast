from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name="importance_temporal",
    layer="7_importance",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="static_snapshot", description="single training-window snapshot", status="operational", priority="A"),
        EnumRegistryEntry(id="time_average", description="time-averaged summary", status="registry_only", priority="B"),
        EnumRegistryEntry(id="rolling_path", description="rolling importance path", status="registry_only", priority="B"),
    ),
    compatible_with={},
    incompatible_with={},
)
