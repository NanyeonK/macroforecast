from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='hp_space_style',
    layer='3_training',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='discrete_grid',
            description='discrete grid',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='continuous_box',
            description='continuous box',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='log_uniform',
            description='log uniform',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='categorical',
            description='categorical',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='conditional_space',
            description='conditional space',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
