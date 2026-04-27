from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='data_richness_mode',
    layer='2_preprocessing',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='target_lags_only',
            description='target lags only',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='factors_plus_target_lags',
            description='factor features plus target lags',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='high_dimensional_predictors',
            description='full high-dimensional predictor panel',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='selected_sparse_predictors',
            description='selected sparse predictor panel',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='mixed_feature_blocks',
            description='mixed feature-block representation',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
    component="feature_representation",
)
