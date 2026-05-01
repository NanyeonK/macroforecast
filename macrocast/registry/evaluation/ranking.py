from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name='ranking',
    layer='4_evaluation',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='by_primary_metric', description='rank by primary metric', status='operational', priority='A'),
        EnumRegistryEntry(id='by_relative_metric', description='rank by relative metric', status='operational', priority='A'),
        EnumRegistryEntry(id='by_average_rank', description='rank by average rank across metrics', status='operational', priority='A'),
        EnumRegistryEntry(id='mcs_inclusion', description='rank by MCS inclusion', status='operational', priority='A'),
    ),
    compatible_with={},
    incompatible_with={},
)
