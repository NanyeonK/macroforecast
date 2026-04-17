from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='separation_rule',
    layer='2_preprocessing',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='strict_separation',
            description='fit preprocessor on X_train only; transform both (no leak)',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='shared_transform_then_split',
            description='fit on combined X then split (intentional leak for replication)',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='X_only_transform',
            description='transform X only, leave y untouched',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='target_only_transform',
            description='transform y only, leave X untouched',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='joint_preprocessor',
            description='user-supplied joint pipeline (paper replication)',
            status='registry_only',
            priority='A',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
