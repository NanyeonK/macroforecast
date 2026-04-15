from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='evaluation_scale',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='raw_level',
            description='raw level',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='transformed_scale',
            description='transformed scale',
            status='registry_only',
            priority='A',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
