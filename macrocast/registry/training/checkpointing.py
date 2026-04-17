from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='checkpointing',
    layer='3_training',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='none',
            description='none',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='per_model',
            description='per model',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='per_horizon',
            description='per horizon',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='per_date',
            description='per date',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='per_trial',
            description='per trial',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
