from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='custom_source_policy',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='official_only',
            description='use the selected official source panel only',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='custom_panel_only',
            description='load a user supplied file instead of the selected official panel',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='official_plus_custom',
            description='append a user supplied file to the selected official panel',
            status='operational',
            priority="A",
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
