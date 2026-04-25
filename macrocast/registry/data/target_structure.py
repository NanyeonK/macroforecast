from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='target_structure',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='single_target_point_forecast',
            description='single target point forecast',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='multi_target_point_forecast',
            description='multi target point forecast',
            status='operational',
            priority="A",
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
