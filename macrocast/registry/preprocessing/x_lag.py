from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='x_lag_creation',
    layer='2_preprocessing',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='no_predictor_lags',
            description='no predictor lags',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='fixed_predictor_lags',
            description='fixed predictor lags',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='cv_selected_predictor_lags',
            description='cross-validation selected predictor lags',
            status='registry_only',
            priority='A',
        ),
        EnumRegistryEntry(
            id='variable_specific_predictor_lags',
            description='variable-specific predictor lags',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='category_specific_predictor_lags',
            description='category-specific predictor lags',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
