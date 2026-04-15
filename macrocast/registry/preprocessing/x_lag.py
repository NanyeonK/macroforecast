from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='x_lag_creation',
    layer='2_preprocessing',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='no_x_lags',
            description='no x lags',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='fixed_x_lags',
            description='fixed x lags',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='cv_selected_x_lags',
            description='cv selected x lags',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='variable_specific_lags',
            description='variable specific lags',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='category_specific_lags',
            description='category specific lags',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
