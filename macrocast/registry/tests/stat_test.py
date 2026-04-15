from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='stat_test',
    layer='6_stat_tests',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='none',
            description='none',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='dm',
            description='dm',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='cw',
            description='cw',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='mcs',
            description='mcs',
            status='planned',
            priority="A",
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
