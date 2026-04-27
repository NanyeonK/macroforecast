from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='benchmark_family',
    layer='3_training',
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
            id='autoregressive_bic',
            description='autoregressive benchmark with BIC lag selection',
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
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='autoregressive_fixed_lag',
            description='autoregressive benchmark with fixed lag count',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='autoregressive_diffusion_index',
            description='autoregressive diffusion index benchmark',
            status='operational',
            priority='B',
        ),
        EnumRegistryEntry(
            id='factor_model_benchmark',
            description='factor model benchmark',
            status='operational',
            priority='B',
        ),
        EnumRegistryEntry(
            id='expert_benchmark',
            description='expert benchmark',
            status='operational',
            priority='B',
        ),
        EnumRegistryEntry(
            id='paper_specific_benchmark',
            description='paper specific benchmark',
            status='operational',
            priority='B',
        ),
        EnumRegistryEntry(
            id='survey_forecast',
            description='survey forecast',
            status='operational',
            priority='B',
        ),
        EnumRegistryEntry(
            id='benchmark_suite',
            description='multiple benchmark suite',
            status='operational',
            priority='A',
        ),
    ),
    compatible_with={},
    incompatible_with={},
    component="benchmark",
)
