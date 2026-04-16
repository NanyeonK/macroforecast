from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name='provenance_fields',
    layer='5_output_provenance',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='none', description='no provenance fields', status='operational', priority='A'),
        EnumRegistryEntry(id='minimal', description='recipe_id, run_id, timestamp only', status='operational', priority='A'),
        EnumRegistryEntry(id='standard', description='standard provenance including git hash', status='operational', priority='A'),
        EnumRegistryEntry(id='full', description='full provenance with config hash and all metadata', status='operational', priority='A'),
    ),
    compatible_with={},
    incompatible_with={},
)
