from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='x_missing_policy',
    layer='2_preprocessing',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='none', description='none', status='operational', priority="A"),
        EnumRegistryEntry(id='drop', description='drop', status='registry_only', priority="A"),
        EnumRegistryEntry(id='em_impute', description='em impute', status='operational', priority="A"),
        EnumRegistryEntry(id='mean_impute', description='mean impute', status='operational', priority="A"),
        EnumRegistryEntry(id='median_impute', description='median impute', status='operational', priority="A"),
        EnumRegistryEntry(id='ffill', description='ffill', status='operational', priority="A"),
        EnumRegistryEntry(id='interpolate_linear', description='interpolate linear', status='operational', priority="A"),
        EnumRegistryEntry(id='drop_rows', description='drop rows', status='planned', priority="A"),
        EnumRegistryEntry(id='drop_columns', description='drop columns', status='planned', priority="A"),
        EnumRegistryEntry(id='drop_if_above_threshold', description='drop if above threshold', status='planned', priority="A"),
        EnumRegistryEntry(id='missing_indicator', description='missing indicator', status='planned', priority="A"),
        EnumRegistryEntry(id='custom', description='custom', status='external_plugin', priority="A"),
    ),
    compatible_with={},
    incompatible_with={},
)
