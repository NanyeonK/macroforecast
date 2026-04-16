from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='point_metrics',
    layer='4_evaluation',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='MSE', description='mse', status='operational', priority='A'),
        EnumRegistryEntry(id='MSFE', description='msfe', status='operational', priority='A'),
        EnumRegistryEntry(id='RMSE', description='rmse', status='operational', priority='A'),
        EnumRegistryEntry(id='MAE', description='mae', status='operational', priority='A'),
        EnumRegistryEntry(id='MAPE', description='mape', status='operational', priority='A'),
        EnumRegistryEntry(id='sMAPE', description='smape', status='registry_only', priority='B'),
        EnumRegistryEntry(id='MASE', description='mase', status='registry_only', priority='B'),
        EnumRegistryEntry(id='RMSSE', description='rmsse', status='registry_only', priority='B'),
        EnumRegistryEntry(id='MedAE', description='medae', status='registry_only', priority='B'),
        EnumRegistryEntry(id='Huber_loss', description='huber loss', status='registry_only', priority='B'),
        EnumRegistryEntry(id='QLIKE', description='qlike', status='registry_only', priority='B'),
        EnumRegistryEntry(id='TheilU', description='theil u', status='registry_only', priority='B'),
    ),
    compatible_with={},
    incompatible_with={},
)
