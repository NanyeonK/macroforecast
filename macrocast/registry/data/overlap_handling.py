from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='overlap_handling',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='allow_overlap',
            description='allow overlap',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='evaluate_with_hac',
            description='evaluate with hac',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='evaluate_with_block_bootstrap',
            description='evaluate with block bootstrap',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='non_overlapping_subsample',
            description='non overlapping subsample',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='horizon_specific_subsample',
            description='horizon specific subsample',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
