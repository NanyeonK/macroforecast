from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name='agg_time',
    layer='4_evaluation',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='full_oos_average', description='full oos average', status='operational', priority='A'),
        EnumRegistryEntry(id='rolling_average', description='rolling average', status='planned', priority='A'),
        EnumRegistryEntry(id='regime_subsample_average', description='regime subsample average', status='planned', priority='A'),
        EnumRegistryEntry(id='event_window_average', description='event window average', status='registry_only', priority='B'),
        EnumRegistryEntry(id='pre_post_break_average', description='pre post break average', status='planned', priority='A'),
    ),
    compatible_with={},
    incompatible_with={},
)
