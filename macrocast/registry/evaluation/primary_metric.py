from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='primary_metric',
    layer='4_evaluation',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='mse', description='mean squared forecast error', status='operational', priority="A"),
        EnumRegistryEntry(id='rmse', description='root mean squared forecast error', status='operational', priority="A"),
        EnumRegistryEntry(id='mae', description='mean absolute forecast error', status='operational', priority="A"),
        EnumRegistryEntry(id='mape', description='mean absolute percentage error', status='operational', priority="B"),
        EnumRegistryEntry(id='medae', description='median absolute error', status='operational', priority="B"),
        EnumRegistryEntry(id='theil_u1', description='Theil U1 statistic', status='operational', priority="B"),
        EnumRegistryEntry(id='theil_u2', description='Theil U2 statistic', status='operational', priority="B"),
        EnumRegistryEntry(id='relative_mse', description='MSE relative to benchmark', status='operational', priority="A"),
        EnumRegistryEntry(id='r2_oos', description='out-of-sample R squared', status='operational', priority="A"),
        EnumRegistryEntry(id='relative_mae', description='MAE relative to benchmark', status='operational', priority="A"),
        EnumRegistryEntry(id='mse_reduction', description='MSE reduction relative to benchmark', status='operational', priority="A"),
        EnumRegistryEntry(id='log_score', description='log predictive score', status='operational', priority="B"),
        EnumRegistryEntry(id='crps', description='continuous ranked probability score', status='operational', priority="B"),
    ),
    compatible_with={},
    incompatible_with={},
)
