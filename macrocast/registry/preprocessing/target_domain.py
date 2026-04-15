from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='target_domain',
    layer='2_preprocessing',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='unconstrained',
            description='unconstrained',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='nonnegative',
            description='nonnegative',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='bounded_0_1',
            description='bounded 0 1',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='integer_count',
            description='integer count',
            status='future',
            priority='B',
        ),
        EnumRegistryEntry(
            id='probability_target',
            description='probability target',
            status='future',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
