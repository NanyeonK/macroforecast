from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='sequence_framework',
    layer='3_training',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='not_sequence',
            description='not sequence',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='fixed_lookback_sequence',
            description='fixed lookback sequence',
            status='future',
            priority='B',
        ),
        EnumRegistryEntry(
            id='variable_lookback_sequence',
            description='variable lookback sequence',
            status='future',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
