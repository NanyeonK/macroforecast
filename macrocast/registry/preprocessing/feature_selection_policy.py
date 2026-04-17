from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='feature_selection_policy',
    layer='2_preprocessing',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='none', description='none', status='operational', priority="A"),
        EnumRegistryEntry(id='correlation_filter', description='correlation filter', status='operational', priority="A"),
        EnumRegistryEntry(id='lasso_select', description='lasso select', status='operational', priority="A"),
        EnumRegistryEntry(id='mutual_information_screen', description='mutual information screen', status='registry_only', priority="A"),
        EnumRegistryEntry(id='custom', description='custom', status='external_plugin', priority="A"),
    ),
    compatible_with={},
    incompatible_with={},
    component="preprocessing",
)
