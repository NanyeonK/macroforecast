from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='tcode_application_scope',
    layer='2_preprocessing',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='apply_tcode_to_target',
            description='t-code on target only',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='apply_tcode_to_X',
            description='t-code on predictors only',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='apply_tcode_to_both',
            description='t-code on target and X',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='apply_tcode_to_none',
            description='no t-code application',
            status='operational',
            priority='A',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
