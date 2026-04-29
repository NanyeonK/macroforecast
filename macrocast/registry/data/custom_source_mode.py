from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='custom_source_mode',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='no_custom_source',
            description='use the selected official source panel only',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='replace_official_panel',
            description='load a user supplied file instead of the selected official panel',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='append_to_official_panel',
            description='append a user supplied file to the selected official panel',
            status='operational',
            priority="A",
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
