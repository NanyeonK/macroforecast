from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name='benchmark_window',
    layer='4_evaluation',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='expanding', description='expanding', status='operational', priority='A'),
        EnumRegistryEntry(id='rolling', description='rolling', status='operational', priority='A'),
        EnumRegistryEntry(id='fixed', description='fixed', status='planned', priority='A'),
        EnumRegistryEntry(id='paper_exact_window', description='paper exact window', status='registry_only', priority='A'),
    ),
    compatible_with={},
    incompatible_with={},
)
