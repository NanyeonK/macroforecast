from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name="importance_stability",
    layer="7_importance",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="importance_stability", description="bootstrap / seed stability summary", status="operational", priority="A"),
        EnumRegistryEntry(id="bootstrap_rank_stability", description="bootstrap rank stability", status="operational", priority="A"),
        EnumRegistryEntry(id="seed_stability", description="seed stability", status="operational", priority="A"),
    ),
    compatible_with={},
    incompatible_with={},
)
