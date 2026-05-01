from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name='benchmark_window',
    layer='4_evaluation',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='full_oos', description='full out-of-sample period', status='operational', priority='A'),
        EnumRegistryEntry(id='rolling', description='rolling', status='operational', priority='A'),
        EnumRegistryEntry(id='expanding', description='expanding', status='operational', priority='A'),
    ),
    compatible_with={},
    incompatible_with={},
)
