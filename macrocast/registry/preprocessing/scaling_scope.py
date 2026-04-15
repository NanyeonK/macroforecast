from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='scaling_scope',
    layer='2_preprocessing',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='columnwise',
            description='columnwise',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='datewise_cross_sectional',
            description='datewise cross sectional',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='groupwise',
            description='groupwise',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='categorywise',
            description='categorywise',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='global_train_only',
            description='global train only',
            status='operational',
            priority='A',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
