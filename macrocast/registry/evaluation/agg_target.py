from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name='agg_target',
    layer='4_evaluation',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='pool_targets', description='pool targets', status='operational', priority='A'),
        EnumRegistryEntry(id='per_target_separate', description='report each target separately', status='operational', priority='A'),
    ),
    compatible_with={},
    incompatible_with={},
)
