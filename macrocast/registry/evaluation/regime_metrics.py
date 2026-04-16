from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name='regime_metrics',
    layer='4_evaluation',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='all_main_metrics_by_regime', description='all main metrics by regime', status='operational', priority='A'),
        EnumRegistryEntry(id='regime_transition_performance', description='regime transition performance', status='registry_only', priority='B'),
        EnumRegistryEntry(id='crisis_period_gain', description='crisis period gain', status='operational', priority='A'),
        EnumRegistryEntry(id='state_dependent_oos_r2', description='state dependent oos r2', status='operational', priority='A'),
    ),
    compatible_with={},
    incompatible_with={},
)
