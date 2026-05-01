from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='point_metrics',
    layer='4_evaluation',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='mse', description='mse', status='operational', priority='A'),
        EnumRegistryEntry(id='rmse', description='rmse', status='operational', priority='A'),
        EnumRegistryEntry(id='mae', description='mae', status='operational', priority='A'),
        EnumRegistryEntry(id='mape', description='mape', status='operational', priority='A'),
        EnumRegistryEntry(id='medae', description='median absolute error', status='operational', priority='B'),
        EnumRegistryEntry(id='theil_u1', description='Theil U1 statistic', status='operational', priority='B'),
        EnumRegistryEntry(id='theil_u2', description='Theil U2 statistic', status='operational', priority='B'),
    ),
    compatible_with={},
    incompatible_with={},
)
