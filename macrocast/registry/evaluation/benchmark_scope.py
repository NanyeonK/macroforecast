from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name='benchmark_scope',
    layer='4_evaluation',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='same_for_all', description='same for all', status='operational', priority='A'),
        EnumRegistryEntry(id='target_specific', description='target specific', status='operational', priority='A'),
        EnumRegistryEntry(id='horizon_specific', description='horizon specific', status='operational', priority='A'),
        EnumRegistryEntry(id='target_horizon_specific', description='target horizon specific', status='registry_only', priority='B'),
    ),
    compatible_with={},
    incompatible_with={},
)
