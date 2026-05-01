from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name='agg_horizon',
    layer='4_evaluation',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='pool_horizons', description='pool forecast horizons', status='operational', priority='A'),
        EnumRegistryEntry(id='per_horizon_separate', description='report each horizon separately', status='operational', priority='A'),
    ),
    compatible_with={},
    incompatible_with={},
)
