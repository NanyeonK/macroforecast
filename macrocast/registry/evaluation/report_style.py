from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name='report_style',
    layer='4_evaluation',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='single_table', description='single model-by-metric table', status='operational', priority='A'),
        EnumRegistryEntry(id='per_target_horizon_panel', description='panel by target and horizon', status='operational', priority='A'),
        EnumRegistryEntry(id='latex_table', description='latex table', status="operational", priority='A'),
    ),
    compatible_with={},
    incompatible_with={},
)
