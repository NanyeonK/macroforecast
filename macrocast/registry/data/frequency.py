from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='frequency',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
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
    ),
    compatible_with={},
    incompatible_with={},
)
