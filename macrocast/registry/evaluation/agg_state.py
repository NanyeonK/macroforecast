from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="agg_state",
    layer="4_evaluation",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="pool_states", description="pool FRED-SD states", status="operational", priority="A"),
        EnumRegistryEntry(id="per_state_separate", description="report each FRED-SD state separately", status="operational", priority="A"),
    ),
    compatible_with={},
    incompatible_with={},
)
