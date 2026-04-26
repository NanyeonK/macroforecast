from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="state_selection",
    layer="1_data_task",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(
            id="all_states",
            description="use all FRED-SD states in the selected workbook variables",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="selected_states",
            description="use leaf_config.sd_states to load only selected FRED-SD state columns",
            status="operational",
            priority="A",
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
