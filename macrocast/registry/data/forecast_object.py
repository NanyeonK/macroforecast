from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='forecast_object',
    layer='1_data_task',
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
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='interval',
            description='interval',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='density',
            description='density',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='direction',
            description='direction',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='turning_point',
            description='turning point',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='regime_probability',
            description='regime probability',
            status='future',
            priority='B',
        ),
        EnumRegistryEntry(
            id='event_probability',
            description='event probability',
            status='future',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
