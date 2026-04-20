from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='economic_metrics',
    layer='4_evaluation',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='utility_gain', description='utility gain', status='future', priority='B'),
        EnumRegistryEntry(id='certainty_equivalent', description='certainty equivalent', status='future', priority='B'),
        EnumRegistryEntry(id='cost_sensitive_loss', description='cost sensitive loss', status='future', priority='B'),
        EnumRegistryEntry(id='policy_loss', description='policy loss', status='future', priority='B'),
        EnumRegistryEntry(id='turning_point_value', description='turning point value', status='future', priority='B'),
    ),
    compatible_with={},
    incompatible_with={},
)
