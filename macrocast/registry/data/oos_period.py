from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='oos_period',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='single_oos_block',
            description='single oos block',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='multiple_oos_blocks',
            description='multiple oos blocks',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='rolling_origin',
            description='rolling origin',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='recession_only_oos',
            description='recession only oos',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='expansion_only_oos',
            description='expansion only oos',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='event_window_oos',
            description='event window oos',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
