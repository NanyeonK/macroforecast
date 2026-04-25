from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='deterministic_components',
    layer='2_preprocessing',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='none',
            description='none',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='constant_only',
            description='constant only',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='linear_trend',
            description='linear trend',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='monthly_seasonal',
            description='monthly seasonal',
            status='operational',
            priority='B',
        ),
        EnumRegistryEntry(
            id='quarterly_seasonal',
            description='quarterly seasonal',
            status='operational',
            priority='B',
        ),
        EnumRegistryEntry(
            id='break_dummies',
            description='break dummies',
            status='operational',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
