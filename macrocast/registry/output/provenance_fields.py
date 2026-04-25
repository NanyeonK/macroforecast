from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name='provenance_fields',
    layer='5_output_provenance',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='none', description='required manifest audit fields only', status='operational', priority='A'),
        EnumRegistryEntry(id='minimal', description='required manifest audit fields without environment hash sidecars', status='operational', priority='A'),
        EnumRegistryEntry(id='standard', description='standard provenance including git hash and package version', status='operational', priority='A'),
        EnumRegistryEntry(id='full', description='standard provenance plus config hash', status='operational', priority='A'),
    ),
    compatible_with={},
    incompatible_with={},
)
