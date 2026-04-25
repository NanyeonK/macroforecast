from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='predictor_family',
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
            id='all_macro_vars',
            description='all macro vars',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='category_based',
            description='category based',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='factor_only',
            description='factor only',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='handpicked_set',
            description='handpicked set',
            status='operational',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
    component="feature_representation",
)
