from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='forecast_object',
    layer='3_training',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='point_mean',
            description='point mean',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='point_median',
            description='point median',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='quantile',
            description='quantile',
            status='operational',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
