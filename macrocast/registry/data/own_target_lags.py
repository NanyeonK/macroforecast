from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='own_target_lags',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='include',
            description='include',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='exclude',
            description='exclude',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='cv_select_lags',
            description='cv select lags',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='target_specific_lag_count',
            description='target specific lag count',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
