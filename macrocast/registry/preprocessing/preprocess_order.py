from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='preprocess_order',
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
            id='tcode_only',
            description='tcode only',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='extra_only',
            description='extra only',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='tcode_then_extra',
            description='tcode then extra',
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
            id='custom',
            description='custom',
            status='external_plugin',
            priority="A",
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
