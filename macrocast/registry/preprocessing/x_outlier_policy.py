from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='x_outlier_policy',
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
            id='clip',
            description='clip',
            status='registry_only',
            priority="A",
        ),
        EnumRegistryEntry(
            id='outlier_to_nan',
            description='outlier to nan',
            status='registry_only',
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
