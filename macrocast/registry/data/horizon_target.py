from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='horizon_target_construction',
    layer='1_data_task',
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
            status='registry_only',
            priority='A',
        ),
        EnumRegistryEntry(
            id='future_logdiff',
            description='future logdiff',
            status='registry_only',
            priority='A',
        ),
        EnumRegistryEntry(
            id='cumulative_growth_to_h',
            description='cumulative growth to h',
            status='registry_only',
            priority='A',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
