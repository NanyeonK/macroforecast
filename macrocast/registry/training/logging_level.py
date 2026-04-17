from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='logging_level',
    layer='3_training',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='silent',
            description='silent',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='info',
            description='info',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='debug',
            description='debug',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='trace',
            description='trace',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
