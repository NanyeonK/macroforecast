from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='factor_count',
    layer='2_preprocessing',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='fixed',
            description='fixed',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='cv_select',
            description='cv select',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='variance_explained_rule',
            description='variance explained rule',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='BaiNg_rule',
            description='BaiNg rule',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='model_specific',
            description='model specific',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
