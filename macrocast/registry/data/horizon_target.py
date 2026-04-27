from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='horizon_target_construction',
    layer='2_preprocessing',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='future_target_level_t_plus_h',
            description='future target level at t plus h',
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
        EnumRegistryEntry(
            id='average_growth_1_to_h',
            description='direct average growth target over steps 1 through h',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='path_average_growth_1_to_h',
            description='Layer 3 executable path-average target built from separate stepwise growth targets',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='average_difference_1_to_h',
            description='direct average difference target over steps 1 through h',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='path_average_difference_1_to_h',
            description='Layer 3 executable path-average target built from separate stepwise difference targets',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='average_log_growth_1_to_h',
            description='direct average log-growth target over steps 1 through h',
            status='operational',
            priority='B',
        ),
        EnumRegistryEntry(
            id='path_average_log_growth_1_to_h',
            description='Layer 3 executable path-average target built from separate stepwise log-growth targets',
            status='operational',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
