from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='target_missing_policy',
    layer='2_preprocessing',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='none',
            description='none',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='drop',
            description='drop',
            status='registry_only',
            priority="A",
        ),
        EnumRegistryEntry(
            id='em_impute',
            description='em impute',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='custom',
            description='custom',
            status='external_plugin',
            priority="A",
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
