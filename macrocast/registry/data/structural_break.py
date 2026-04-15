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
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='pre_post_covid',
            description='pre post covid',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='user_break_dates',
            description='user break dates',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='break_test_detected',
            description='break test detected',
            status='future',
            priority='B',
        ),
        EnumRegistryEntry(
            id='rolling_break_adaptive',
            description='rolling break adaptive',
            status='future',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
