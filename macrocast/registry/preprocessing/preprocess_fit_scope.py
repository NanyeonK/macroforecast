from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='preprocess_fit_scope',
    layer='2_preprocessing',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='not_applicable',
            description='not applicable',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='train_only',
            description='train only',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='expanding_train_only',
            description='expanding train only',
            status='registry_only',
            priority="A",
        ),
        EnumRegistryEntry(
            id='rolling_train_only',
            description='rolling train only',
            status='registry_only',
            priority="A",
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
