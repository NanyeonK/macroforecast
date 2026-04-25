from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name="importance_aggregation",
    layer="7_importance",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="mean_abs", description="mean absolute contribution", status="operational", priority="A"),
        EnumRegistryEntry(id="mean_signed", description="mean signed contribution", status="registry_only", priority="B"),
        EnumRegistryEntry(id="top_k", description="top-k ranked summary", status="registry_only", priority="B"),
    ),
    compatible_with={},
    incompatible_with={},
)
