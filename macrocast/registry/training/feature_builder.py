from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='feature_builder',
    layer='2_preprocessing',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='autoreg_lagged_target', description='autoreg lagged target', status='operational', priority="A"),
        EnumRegistryEntry(id='factors_plus_AR', description='factors plus AR', status='operational', priority="A"),
        EnumRegistryEntry(id='raw_feature_panel', description='raw feature panel', status='operational', priority="A"),
        EnumRegistryEntry(id='raw_X_only', description='raw X only', status='operational', priority="A"),
        EnumRegistryEntry(id='factor_pca', description='factor pca', status='operational', priority="A"),
        EnumRegistryEntry(id='sequence_tensor', description='sequence tensor', status='future', priority="B"),
    ),
    compatible_with={},
    incompatible_with={},
    component="feature_builder",
)
