from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='representation_policy',
    layer='2_preprocessing',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='raw_only',
            description='no representation transform',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='tcode_only',
            description='t-code transform only, no extra preprocess',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='custom_transform_only',
            description='user-defined representation',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
