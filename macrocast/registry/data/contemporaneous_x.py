from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='contemporaneous_x_rule',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='allow_contemporaneous',
            description='allow contemporaneous',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='forbid_contemporaneous',
            description='forbid contemporaneous',
            status='operational',
            priority='A',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
