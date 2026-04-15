from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='contemporaneous_x_rule',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='allow_contemporaneous',
            description='allow contemporaneous',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='forbid_contemporaneous',
            description='forbid contemporaneous',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='allow_if_available_in_real_time',
            description='allow if available in real time',
            status='future',
            priority='B',
        ),
        EnumRegistryEntry(
            id='series_specific_contemporaneous',
            description='series specific contemporaneous',
            status='future',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
