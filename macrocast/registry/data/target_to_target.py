from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='target_to_target_inclusion',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='allow_other_targets_as_X',
            description='allow other targets as X',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='forbid_other_targets_as_X',
            description='forbid other targets as X',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='allow_selected_targets_as_X',
            description='allow selected targets as X',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='system_wide_joint_model',
            description='system wide joint model',
            status='future',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
