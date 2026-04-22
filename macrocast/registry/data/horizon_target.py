from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='horizon_target_construction',
    layer='2_preprocessing',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='future_level_y_t_plus_h',
            description='future level y t plus h',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='future_diff',
            description='future diff',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='future_logdiff',
            description='future logdiff',
            status='operational',
            priority='A',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
