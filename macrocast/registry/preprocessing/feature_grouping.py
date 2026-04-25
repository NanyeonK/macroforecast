from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='feature_grouping',
    layer='2_preprocessing',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='none',
            description='none',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='fred_category_group',
            description='fred category group',
            status='registry_only',
            priority='A',
        ),
        EnumRegistryEntry(
            id='economic_theme_group',
            description='economic theme group',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='lag_group',
            description='lag group',
            status='registry_only',
            priority='A',
        ),
        EnumRegistryEntry(
            id='factor_group',
            description='factor group',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
    component="feature_representation",
)
