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
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='future_logdiff',
            description='future logdiff',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='cumulative_growth_to_h',
            description='cumulative growth to h',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='average_growth_1_to_h',
            description='average growth 1 to h',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='annualized_growth_to_h',
            description='annualized growth to h',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='realized_future_average',
            description='realized future average',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='future_sum',
            description='future sum',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='future_indicator',
            description='future indicator',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
