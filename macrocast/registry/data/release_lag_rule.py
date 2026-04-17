from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='release_lag_rule',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='ignore_release_lag',
            description='ignore release lag (use revised values at forecast origin)',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='fixed_lag_all_series',
            description='apply fixed publication lag (e.g. 1 period) to all series',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='series_specific_lag',
            description='apply per-series publication lag from lag table',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='calendar_exact_lag',
            description='apply calendar-exact publication date for each release',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='lag_conservative',
            description='conservative larger publication lag',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='lag_aggressive',
            description='aggressive smaller publication lag',
            status='operational',
            priority='A',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
