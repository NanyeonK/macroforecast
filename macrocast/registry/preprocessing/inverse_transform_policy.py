from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='inverse_transform_policy',
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
            id='target_only',
            description='contract-defined inverse transform of target-side predictions before evaluation',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='forecast_scale_only',
            description='contract-defined inverse transform for forecast output scale only',
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
