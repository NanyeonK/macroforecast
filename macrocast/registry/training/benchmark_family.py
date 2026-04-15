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
            priority="A",
        ),
        EnumRegistryEntry(
            id='ar_bic',
            description='ar bic',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='zero_change',
            description='zero change',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='custom_benchmark',
            description='custom benchmark',
            status='operational',
            priority="A",
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
