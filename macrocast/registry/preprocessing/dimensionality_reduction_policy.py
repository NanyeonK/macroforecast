from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='dimensionality_reduction_policy',
    layer='2_preprocessing',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='none', description='none', status='operational', priority="A"),
        EnumRegistryEntry(id='pca', description='pca', status='operational', priority="A"),
        EnumRegistryEntry(id='static_factor', description='static factor', status='operational', priority="A"),
        EnumRegistryEntry(id='ipca', description='ipca', status='planned', priority="A"),
        EnumRegistryEntry(id='custom', description='custom', status='external_plugin', priority="A"),
    ),
    compatible_with={},
    incompatible_with={},
)
