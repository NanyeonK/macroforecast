from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='benchmark_family',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='historical_mean',
            description='historical mean',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='ar_bic',
            description='ar bic',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='zero_change',
            description='zero change',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='custom_benchmark',
            description='custom benchmark',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='rolling_mean',
            description='rolling mean',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='random_walk',
            description='random walk',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='ar_fixed_p',
            description='ar fixed p',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='ardi',
            description='ardi',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='factor_model',
            description='factor model',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='var',
            description='var',
            status='future',
            priority='B',
        ),
        EnumRegistryEntry(
            id='expert_benchmark',
            description='expert benchmark',
            status='future',
            priority='B',
        ),
        EnumRegistryEntry(
            id='paper_specific_benchmark',
            description='paper specific benchmark',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
