from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='alignment_fairness',
    layer='3_training',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='same_split_across_models',
            description='same split across models',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='same_split_across_targets',
            description='same split across targets',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='same_split_across_horizons',
            description='same split across horizons',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='model_specific_split_allowed',
            description='model specific split allowed',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='target_specific_split_allowed',
            description='target specific split allowed',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
