from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='alignment_rule',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='end_of_period',
            description='end of period',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='average_within_period',
            description='average within period',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='last_available',
            description='last available',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='quarter_to_month_repeat',
            description='quarter to month repeat',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='month_to_quarter_average',
            description='month to quarter average',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='month_to_quarter_last',
            description='month to quarter last',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='ragged_edge_keep',
            description='ragged edge keep',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='ragged_edge_fill',
            description='ragged edge fill',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='calendar_strict',
            description='calendar strict',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
