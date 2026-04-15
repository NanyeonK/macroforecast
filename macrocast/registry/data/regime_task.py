from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='regime_task',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='unconditional',
            description='unconditional',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='recession_conditioned',
            description='recession conditioned',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='expansion_conditioned',
            description='expansion conditioned',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='high_inflation_conditioned',
            description='high inflation conditioned',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='user_defined_regime',
            description='user defined regime',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
