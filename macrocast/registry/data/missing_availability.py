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
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='target_date_drop_if_missing',
            description='drop dates where target value is missing',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='x_impute_only',
            description='impute X only, drop rows where target missing',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='real_time_missing_as_missing',
            description='preserve real-time missingness',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='state_space_fill',
            description='state-space (Kalman) imputation',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='factor_fill',
            description='factor-model based imputation',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='em_fill',
            description='EM algorithm imputation',
            status='operational',
            priority='A',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
