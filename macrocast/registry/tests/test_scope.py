from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="test_scope",
    layer="6_stat_tests",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="per_target_horizon", description="Run tests per target-horizon cell", status="operational", priority="A"),
        EnumRegistryEntry(id="per_target", description="Run tests once per target", status="operational", priority="A"),
        EnumRegistryEntry(id="per_horizon", description="Run tests once per horizon", status="operational", priority="A"),
        EnumRegistryEntry(id="pooled", description="Pool eligible cells before testing", status="operational", priority="A"),
    ),
    compatible_with={},
    incompatible_with={},
)
