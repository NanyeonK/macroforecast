from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='target_family',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='single_macro_series',
            description='single macro series',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='multiple_macro_series',
            description='multiple macro series',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='panel_target',
            description='panel target',
            status='future',
            priority='B',
        ),
        EnumRegistryEntry(
            id='state_target',
            description='state target',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='factor_target',
            description='factor target',
            status='future',
            priority='B',
        ),
        EnumRegistryEntry(
            id='latent_target',
            description='latent target',
            status='future',
            priority='B',
        ),
        EnumRegistryEntry(
            id='constructed_target',
            description='constructed target',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='classification_target',
            description='classification target',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
