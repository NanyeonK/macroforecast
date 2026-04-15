from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='warmup_rule',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='lags_only_warmup',
            description='lags only warmup',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='lags_and_factors_warmup',
            description='lags and factors warmup',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='sequence_warmup',
            description='sequence warmup',
            status='future',
            priority='B',
        ),
        EnumRegistryEntry(
            id='transform_warmup',
            description='transform warmup',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='indicator_warmup',
            description='indicator warmup',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
