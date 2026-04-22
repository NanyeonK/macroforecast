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
            id='factor_plus_lags',
            description='factor plus lags',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='full_high_dimensional_X',
            description='full high dimensional X',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='selected_sparse_X',
            description='selected sparse X',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='mixed_mode',
            description='mixed mode',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
