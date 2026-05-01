from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name='decomposition_target',
    layer='4_evaluation',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='none', description='do not decompose evaluation metrics', status='operational', priority='A'),
        EnumRegistryEntry(id='by_horizon', description='decompose by forecast horizon', status='operational', priority='A'),
        EnumRegistryEntry(id='by_target', description='decompose by target', status='operational', priority='A'),
        EnumRegistryEntry(id='by_state', description='decompose by FRED-SD state', status='operational', priority='A'),
        EnumRegistryEntry(id='by_regime', description='decompose by regime', status='operational', priority='A'),
    ),
    compatible_with={},
    incompatible_with={},
)
