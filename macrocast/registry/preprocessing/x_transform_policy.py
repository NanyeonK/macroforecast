from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='x_transform_policy',
    layer='2_preprocessing',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='raw_level',
            description='raw level',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='dataset_tcode_transformed',
            description='dataset tcode transformed',
            status='registry_only',
            priority="A",
        ),
        EnumRegistryEntry(
            id='custom_x_transform',
            description='custom x transform',
            status='external_plugin',
            priority="A",
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
