from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='oos_period',
    layer='4_evaluation',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='full_oos', description='full out-of-sample period', status='operational', priority='A'),
        EnumRegistryEntry(id='fixed_dates', description='fixed OOS start/end dates', status='operational', priority='A'),
        EnumRegistryEntry(id='multiple_subperiods', description='multiple named OOS subperiods', status='operational', priority='A'),
    ),
    compatible_with={},
    incompatible_with={},
)
