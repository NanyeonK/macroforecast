from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='scaling_policy',
    layer='2_preprocessing',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='none', description='none', status='operational', priority="A"),
        EnumRegistryEntry(id='standard', description='standard', status='operational', priority="A"),
        EnumRegistryEntry(id='robust', description='robust', status='operational', priority="A"),
        EnumRegistryEntry(id='minmax', description='minmax', status='operational', priority="A"),
        EnumRegistryEntry(id='demean_only', description='demean only', status='operational', priority="A"),
        EnumRegistryEntry(id='unit_variance_only', description='unit variance only', status='operational', priority="A"),
        EnumRegistryEntry(id='rank_scale', description='rank scale', status='registry_only', priority="A"),
        EnumRegistryEntry(id='custom', description='custom', status='external_plugin', priority="A"),
    ),
    compatible_with={},
    incompatible_with={},
    component="preprocessing",
)
