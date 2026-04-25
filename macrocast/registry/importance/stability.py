from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name="importance_stability",
    layer="7_importance",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="none", description="no importance stability summary selected", status="operational", priority="A"),
        EnumRegistryEntry(id="importance_stability", description="bootstrap / seed stability summary", status="operational", priority="A"),
        EnumRegistryEntry(id="bootstrap_rank_stability", description="bootstrap rank stability", status="registry_only", priority="B"),
        EnumRegistryEntry(id="seed_stability", description="seed stability", status="registry_only", priority="B"),
    ),
    compatible_with={},
    incompatible_with={},
)
