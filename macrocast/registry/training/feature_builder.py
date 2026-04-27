from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='feature_builder',
    layer='2_preprocessing',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='target_lag_features', description='target-lag feature panel', status='operational', priority="A"),
        EnumRegistryEntry(id='factors_plus_target_lags', description='factor features plus target lags', status='operational', priority="A"),
        EnumRegistryEntry(id='raw_feature_panel', description='raw feature panel', status='operational', priority="A"),
        EnumRegistryEntry(id='raw_predictors_only', description='raw predictor panel without target lags', status='operational', priority="A"),
        EnumRegistryEntry(id='pca_factor_features', description='PCA factor feature panel', status='operational', priority="A"),
        EnumRegistryEntry(id='sequence_tensor', description='sequence tensor', status='future', priority="B"),
    ),
    compatible_with={},
    incompatible_with={},
    component="feature_representation",
)
