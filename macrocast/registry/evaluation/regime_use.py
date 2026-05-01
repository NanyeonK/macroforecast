from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name='regime_use',
    layer='4_evaluation',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='pooled', description='ignore regimes in evaluation', status='operational', priority='A'),
        EnumRegistryEntry(id='per_regime', description='evaluate separately per regime', status='operational', priority='A'),
        EnumRegistryEntry(id='both', description='report pooled and per-regime metrics', status='operational', priority='A'),
    ),
    compatible_with={},
    incompatible_with={},
)
