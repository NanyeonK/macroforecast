from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='convergence_handling',
    layer='3_training',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='retry_new_seed',
            description='retry new seed',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='clip_grad_and_retry',
            description='clip grad and retry',
            status='future',
            priority='B',
        ),
        EnumRegistryEntry(
            id='fallback_to_safe_hp',
            description='fallback to safe hp',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='mark_fail',
            description='mark fail',
            status='operational',
            priority='A',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
