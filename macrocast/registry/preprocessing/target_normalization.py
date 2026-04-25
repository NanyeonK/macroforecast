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
            description='train-window z-score normalization fit separately inside each OOS training window',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='robust_zscore',
            description='train-window median/MAD robust z-score normalization fit separately inside each OOS training window',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='minmax',
            description='train-window min-max target normalization',
            status='operational',
            priority='B',
        ),
        EnumRegistryEntry(
            id='unit_variance',
            description='train-window unit-variance target normalization without demeaning',
            status='operational',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
