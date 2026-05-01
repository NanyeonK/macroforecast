from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="overlap_handling",
    layer="6_stat_tests",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="nw_with_h_minus_1_lag", description="Newey-West with h-1 lags for overlapping forecast errors", status="operational", priority="A"),
        EnumRegistryEntry(id="west_1996_adjustment", description="West (1996) overlap adjustment", status="operational", priority="A"),
        EnumRegistryEntry(id="none", description="No overlap adjustment", status="operational", priority="A"),
    ),
    compatible_with={},
    incompatible_with={},
)
