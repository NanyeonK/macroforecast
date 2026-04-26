from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="sd_variable_selection",
    layer="1_data_task",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(
            id="all_sd_variables",
            description="load all FRED-SD workbook variable sheets",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="selected_sd_variables",
            description="use leaf_config.sd_variables to load only selected FRED-SD workbook variable sheets",
            status="operational",
            priority="A",
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
