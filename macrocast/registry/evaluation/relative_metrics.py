from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='relative_metrics',
    layer='4_evaluation',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='relative_mse', description='relative mse', status='operational', priority='A'),
        EnumRegistryEntry(id='r2_oos', description='out-of-sample R squared', status='operational', priority='A'),
        EnumRegistryEntry(id='relative_mae', description='relative mae', status='operational', priority='A'),
        EnumRegistryEntry(id='mse_reduction', description='mse reduction relative to benchmark', status='operational', priority='A'),
    ),
    compatible_with={},
    incompatible_with={},
)
