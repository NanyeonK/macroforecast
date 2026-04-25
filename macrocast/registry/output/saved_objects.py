from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name='saved_objects',
    layer='5_output_provenance',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='none', description='no artifacts saved', status='registry_only', priority='B'),
        EnumRegistryEntry(id='predictions_only', description='predictions only', status='operational', priority='A'),
        EnumRegistryEntry(id='predictions_and_metrics', description='predictions and metrics', status='operational', priority='A'),
        EnumRegistryEntry(id='full_bundle', description='all artifacts including models and data', status='operational', priority='A'),
        EnumRegistryEntry(id='models_only', description='model artifacts only', status="future", priority='B'),
        EnumRegistryEntry(id='data_only', description='processed data artifacts only', status="future", priority='B'),
    ),
    compatible_with={},
    incompatible_with={},
)
