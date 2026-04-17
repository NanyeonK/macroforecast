from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='evaluation_scale',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='original_scale',
            description='evaluate metrics on original (untransformed) scale',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='transformed_scale',
            description='evaluate metrics on transformed scale',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='both',
            description='compute metrics on both original and transformed scales',
            status='operational',
            priority='A',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
