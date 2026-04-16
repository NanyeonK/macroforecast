from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name='ranking',
    layer='4_evaluation',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='mean_metric_rank', description='mean metric rank', status='planned', priority='A'),
        EnumRegistryEntry(id='median_metric_rank', description='median metric rank', status='planned', priority='A'),
        EnumRegistryEntry(id='win_count', description='win count', status='planned', priority='A'),
        EnumRegistryEntry(id='benchmark_beat_freq', description='benchmark beat freq', status='planned', priority='A'),
        EnumRegistryEntry(id='MCS_inclusion_priority', description='mcs inclusion priority', status='planned', priority='A'),
        EnumRegistryEntry(id='stability_weighted_rank', description='stability weighted rank', status='registry_only', priority='B'),
        EnumRegistryEntry(id='ensemble_selection_rank', description='ensemble selection rank', status='future', priority='B'),
    ),
    compatible_with={},
    incompatible_with={},
)
