from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='frequency',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='daily',
            description='daily',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='weekly',
            description='weekly',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='monthly',
            description='monthly',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='quarterly',
            description='quarterly',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='yearly',
            description='yearly',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='mixed_frequency',
            description='mixed frequency',
            status='future',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
