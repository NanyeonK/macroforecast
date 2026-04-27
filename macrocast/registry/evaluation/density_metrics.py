from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='density_metrics',
    layer='4_evaluation',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='pinball_loss', description='pinball loss', status='registry_only', priority='B'),
        EnumRegistryEntry(id='crps', description='crps', status='registry_only', priority='B'),
        EnumRegistryEntry(id='interval_score', description='interval score', status='registry_only', priority='B'),
        EnumRegistryEntry(id='coverage_rate', description='coverage rate', status='registry_only', priority='B'),
        EnumRegistryEntry(id='winkler_score', description='winkler score', status='registry_only', priority='B'),
        EnumRegistryEntry(id='log_score', description='log score', status='future', priority='B'),
        EnumRegistryEntry(id='nll', description='nll', status='future', priority='B'),
        EnumRegistryEntry(id='pit_based_metric', description='pit based metric', status='future', priority='B'),
    ),
    compatible_with={},
    incompatible_with={},
)
