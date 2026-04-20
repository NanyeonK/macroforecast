from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='forecast_type',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='direct',
            description='direct',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='iterated',
            description='iterated',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='dirrec',
            description='dirrec',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='mimo',
            description='mimo',
            status='future',
            priority='B',
        ),
        EnumRegistryEntry(
            id='seq2seq',
            description='seq2seq',
            status='future',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
