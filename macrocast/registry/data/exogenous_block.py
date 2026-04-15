from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='exogenous_block',
    layer='1_data_task',
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
            id='strict_exogenous_only',
            description='strict exogenous only',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='endogenous_allowed',
            description='endogenous allowed',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='group_exogenous_blocks',
            description='group exogenous blocks',
            status='future',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
