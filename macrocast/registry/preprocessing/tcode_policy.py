from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='tcode_policy',
    layer='2_preprocessing',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='raw_only',
            description='raw only',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='tcode_only',
            description='tcode only',
            status='registry_only',
            priority="A",
        ),
        EnumRegistryEntry(
            id='tcode_then_extra_preprocess',
            description='tcode then extra preprocess',
            status='registry_only',
            priority="A",
        ),
        EnumRegistryEntry(
            id='extra_preprocess_without_tcode',
            description='extra preprocess without tcode',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='extra_then_tcode',
            description='extra then tcode',
            status='registry_only',
            priority="A",
        ),
        EnumRegistryEntry(
            id='custom_transform_pipeline',
            description='custom transform pipeline',
            status='external_plugin',
            priority="A",
        ),
    ),
    compatible_with={},
    incompatible_with={},
    component="preprocessing",
)
