from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='feature_builder',
    layer='3_training',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='autoreg_lagged_target',
            description='autoreg lagged target',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='raw_feature_panel',
            description='raw feature panel',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='factor_pca',
            description='factor pca',
            status='planned',
            priority="A",
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
