from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='structural_break_segmentation',
    layer='1_data_task',
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
            id='pre_post_crisis',
            description='pre post crisis',
            status='registry_only',
            priority='A',
        ),
        EnumRegistryEntry(
            id='pre_post_covid',
            description='pre post covid',
            status='registry_only',
            priority='A',
        ),
        EnumRegistryEntry(
            id='user_break_dates',
            description='user break dates',
            status='registry_only',
            priority='A',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
