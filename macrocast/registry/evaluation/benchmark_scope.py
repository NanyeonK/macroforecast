from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name='benchmark_scope',
    layer='4_evaluation',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='all_targets_horizons', description='single benchmark across all targets and horizons', status='operational', priority='A'),
        EnumRegistryEntry(id='per_target_horizon', description='benchmark separately per target-horizon cell', status='operational', priority='A'),
    ),
    compatible_with={},
    incompatible_with={},
)
