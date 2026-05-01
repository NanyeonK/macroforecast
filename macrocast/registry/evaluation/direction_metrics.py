from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='direction_metrics',
    layer='4_evaluation',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='success_ratio', description='directional success ratio', status='operational', priority='A'),
        EnumRegistryEntry(id='pesaran_timmermann_metric', description='Pesaran-Timmermann direction statistic value', status='operational', priority='A'),
    ),
    compatible_with={},
    incompatible_with={},
)
