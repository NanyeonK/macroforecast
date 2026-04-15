from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='preprocessing_axis_role',
    layer='2_preprocessing',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='fixed_preprocessing',
            description='preprocessing fixed for fair comparison',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='swept_preprocessing',
            description='preprocessing intentionally varied',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='ablation_preprocessing',
            description='preprocessing part of ablation study',
            status='planned',
            priority='A',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
