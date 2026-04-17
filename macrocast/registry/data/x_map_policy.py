from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='x_map_policy',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='shared_X',
            description='shared X',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='target_specific_X',
            description='target specific X',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='category_specific_X',
            description='category specific X',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='learned_target_X',
            description='learned target X',
            status='future',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
