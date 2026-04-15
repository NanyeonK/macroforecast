from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='target_normalization',
    layer='2_preprocessing',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='none',
            description='none',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='zscore_train_only',
            description='zscore train only',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='robust_zscore',
            description='robust zscore',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='minmax',
            description='minmax',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='unit_variance',
            description='unit variance',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
