from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name='decomposition_order',
    layer='4_evaluation',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='marginal', description='marginal decomposition', status='operational', priority='A'),
        EnumRegistryEntry(id='sequential', description='sequential decomposition', status='operational', priority='A'),
    ),
    compatible_with={},
    incompatible_with={},
)
