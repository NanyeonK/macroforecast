from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='relative_metrics',
    layer='4_evaluation',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='relative_msfe', description='relative msfe', status='operational', priority='A'),
        EnumRegistryEntry(id='relative_rmse', description='relative rmse', status='operational', priority='A'),
        EnumRegistryEntry(id='relative_mae', description='relative mae', status='operational', priority='A'),
        EnumRegistryEntry(id='oos_r2', description='oos r2', status='operational', priority='A'),
        EnumRegistryEntry(id='benchmark_win_rate', description='benchmark win rate', status='operational', priority='A'),
        EnumRegistryEntry(id='csfe_difference', description='csfe difference', status='operational', priority='A'),
    ),
    compatible_with={},
    incompatible_with={},
)
