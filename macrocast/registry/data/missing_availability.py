from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='missing_availability',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='complete_case_only',
            description='drop rows with any missing values',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='available_case',
            description='keep rows with available cases only (per-series)',
            status='registry_only',
            priority='A',
        ),
        EnumRegistryEntry(
            id='x_impute_only',
            description='impute X only, drop rows where target missing',
            status='registry_only',
            priority='A',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
