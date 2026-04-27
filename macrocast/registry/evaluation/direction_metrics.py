from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='direction_metrics',
    layer='4_evaluation',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='directional_accuracy', description='directional accuracy', status='operational', priority='A'),
        EnumRegistryEntry(id='sign_accuracy', description='sign accuracy', status='operational', priority='A'),
        EnumRegistryEntry(id='turning_point_accuracy', description='turning point accuracy', status='registry_only', priority='B'),
        EnumRegistryEntry(id='precision', description='precision', status='registry_only', priority='B'),
        EnumRegistryEntry(id='recall', description='recall', status='registry_only', priority='B'),
        EnumRegistryEntry(id='f1', description='f1', status='registry_only', priority='B'),
        EnumRegistryEntry(id='balanced_accuracy', description='balanced accuracy', status='registry_only', priority='B'),
        EnumRegistryEntry(id='auc', description='auc', status='registry_only', priority='B'),
        EnumRegistryEntry(id='brier_score', description='brier score', status='registry_only', priority='B'),
    ),
    compatible_with={},
    incompatible_with={},
)
