from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name='decomposition_order',
    layer='4_evaluation',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='marginal_effect_only', description='marginal effect only', status='operational', priority='A'),
        EnumRegistryEntry(id='two_way_interaction', description='two way interaction', status='registry_only', priority='B'),
        EnumRegistryEntry(id='three_way_interaction', description='three way interaction', status='future', priority='B'),
        EnumRegistryEntry(id='full_factorial', description='full factorial', status='future', priority='B'),
        EnumRegistryEntry(id='shapley_style_effect_decomp', description='shapley style effect decomp', status='future', priority='B'),
    ),
    compatible_with={},
    incompatible_with={},
)
