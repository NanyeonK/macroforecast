from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='overlap_handling',
    layer='6_stat_tests',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='allow_overlap',
            description='allow overlap (default)',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='evaluate_with_hac',
            description='evaluate with hac covariance in stat tests',
            status='operational',
            priority='A',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
