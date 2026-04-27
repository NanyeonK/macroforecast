from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name='regime_use',
    layer='4_evaluation',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='evaluation_only', description='eval only', status='operational', priority='A'),
        EnumRegistryEntry(id='train_only', description='train only', status='registry_only', priority='B'),
        EnumRegistryEntry(id='train_and_eval', description='train and eval', status='registry_only', priority='B'),
        EnumRegistryEntry(id='regime_specific_model', description='regime specific model', status='future', priority='B'),
        EnumRegistryEntry(id='regime_interaction_features', description='regime interaction features', status='future', priority='B'),
    ),
    compatible_with={},
    incompatible_with={},
)
