from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name='regime_definition',
    layer='4_evaluation',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='none', description='none', status='operational', priority='A'),
        EnumRegistryEntry(id='nber_recession', description='nber recession', status='operational', priority='A'),
        EnumRegistryEntry(id='quantile_uncertainty', description='quantile uncertainty', status='registry_only', priority='B'),
        EnumRegistryEntry(id='financial_stress', description='financial stress', status='registry_only', priority='B'),
        EnumRegistryEntry(id='volatility_regime', description='volatility regime', status='registry_only', priority='B'),
        EnumRegistryEntry(id='markov_switching_regime', description='markov switching regime', status='future', priority='B'),
        EnumRegistryEntry(id='clustering_regime', description='clustering regime', status='future', priority='B'),
        EnumRegistryEntry(id='user_defined_regime', description='user defined regime', status='operational', priority='A'),
    ),
    compatible_with={},
    incompatible_with={},
)
