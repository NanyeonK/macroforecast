from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='vintage_policy',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='latest_only',
            description='latest only',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='single_vintage',
            description='single vintage',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='rolling_vintage',
            description='rolling vintage',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='all_vintages_available',
            description='all vintages available',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='event_vintage_subset',
            description='event vintage subset',
            status='future',
            priority='B',
        ),
        EnumRegistryEntry(
            id='vintage_range',
            description='vintage range',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
