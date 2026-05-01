from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='density_metrics',
    layer='4_evaluation',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='log_score', description='log predictive score', status='operational', priority='A'),
        EnumRegistryEntry(id='crps', description='continuous ranked probability score', status='operational', priority='A'),
        EnumRegistryEntry(id='interval_score', description='interval score', status='operational', priority='B'),
        EnumRegistryEntry(id='coverage_rate', description='coverage rate', status='operational', priority='B'),
    ),
    compatible_with={},
    incompatible_with={},
)
