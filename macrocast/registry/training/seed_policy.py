from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='seed_policy',
    layer='3_training',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='fixed_seed',
            description='fixed seed',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='multi_seed_average',
            description='multi seed average',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='seed_sweep',
            description='seed sweep',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='deterministic_only',
            description='deterministic only',
            status='operational',
            priority='A',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
