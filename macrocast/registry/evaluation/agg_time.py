from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name='agg_time',
    layer='4_evaluation',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='mean', description='mean across OOS periods', status='operational', priority='A'),
        EnumRegistryEntry(id='median', description='median across OOS periods', status='operational', priority='A'),
        EnumRegistryEntry(id='per_subperiod', description='report each subperiod separately', status='operational', priority='A'),
    ),
    compatible_with={},
    incompatible_with={},
)
