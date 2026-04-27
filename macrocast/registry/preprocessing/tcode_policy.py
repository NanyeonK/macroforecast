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
            id='official_tcode_only',
            description='tcode only',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='official_tcode_then_extra_preprocess',
            description='tcode then extra preprocess',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='extra_preprocess_only',
            description='extra preprocess without tcode',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='extra_preprocess_then_official_tcode',
            description='extra then tcode',
            status='registry_only',
            priority="A",
        ),
        EnumRegistryEntry(
            id='custom_transform_sequence',
            description='custom transform pipeline',
            status='external_plugin',
            priority="A",
        ),
    ),
    compatible_with={},
    incompatible_with={},
    component="preprocessing",
)
