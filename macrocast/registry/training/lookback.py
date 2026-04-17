from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='lookback',
    layer='3_training',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='fixed_lookback',
            description='fixed lookback',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='horizon_specific_lookback',
            description='horizon specific lookback',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='target_specific_lookback',
            description='target specific lookback',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='cv_select_lookback',
            description='cv select lookback',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
