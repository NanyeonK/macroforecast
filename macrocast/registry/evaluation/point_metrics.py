from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='point_metrics',
    layer='4_evaluation',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='mse', description='mse', status='operational', priority='A'),
        EnumRegistryEntry(id='msfe', description='msfe', status='operational', priority='A'),
        EnumRegistryEntry(id='rmse', description='rmse', status='operational', priority='A'),
        EnumRegistryEntry(id='mae', description='mae', status='operational', priority='A'),
        EnumRegistryEntry(id='mape', description='mape', status='operational', priority='A'),
        EnumRegistryEntry(id='smape', description='smape', status='registry_only', priority='B'),
        EnumRegistryEntry(id='mase', description='mase', status='registry_only', priority='B'),
        EnumRegistryEntry(id='rmsse', description='rmsse', status='registry_only', priority='B'),
        EnumRegistryEntry(id='median_absolute_error', description='medae', status='registry_only', priority='B'),
        EnumRegistryEntry(id='huber_loss', description='huber loss', status='registry_only', priority='B'),
        EnumRegistryEntry(id='qlike', description='qlike', status='registry_only', priority='B'),
        EnumRegistryEntry(id='theil_u', description='theil u', status='registry_only', priority='B'),
    ),
    compatible_with={},
    incompatible_with={},
)
