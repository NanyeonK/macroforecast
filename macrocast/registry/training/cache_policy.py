from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='cache_policy',
    layer='3_training',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='no_cache',
            description='no cache',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='data_cache',
            description='data cache',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='feature_cache',
            description='feature cache',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='fold_cache',
            description='fold cache',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='prediction_cache',
            description='prediction cache',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
