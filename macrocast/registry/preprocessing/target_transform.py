from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='target_transform',
    layer='2_preprocessing',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='level',
            description='level',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='difference',
            description='difference',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='log',
            description='log',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='log_difference',
            description='log difference',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='growth_rate',
            description='growth rate',
            status='planned',
            priority='A',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
