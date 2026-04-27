from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name='agg_target',
    layer='4_evaluation',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='equal_weight', description='equal weight', status='operational', priority='A'),
        EnumRegistryEntry(id='scale_adjusted_weighting', description='scale adjusted weight', status='registry_only', priority='B'),
        EnumRegistryEntry(id='economic_priority_weighting', description='economic priority weight', status='future', priority='B'),
        EnumRegistryEntry(id='report_separately_only', description='report separately only', status='operational', priority='A'),
    ),
    compatible_with={},
    incompatible_with={},
)
