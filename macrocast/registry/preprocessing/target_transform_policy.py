from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='target_transform_policy',
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
            id='tcode_transformed',
            description='tcode transformed',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='custom_target_transform',
            description='custom target transform',
            status='external_plugin',
            priority="A",
        ),
    ),
    compatible_with={},
    incompatible_with={},
    component="preprocessing",
)
