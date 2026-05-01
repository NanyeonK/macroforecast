from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name='artifact_granularity',
    layer='5_output_provenance',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='per_cell', description='one artifact set per cell', status='operational', priority='A'),
        EnumRegistryEntry(id='per_target', description='one artifact set per target', status='operational', priority='A'),
        EnumRegistryEntry(id='per_horizon', description='one artifact set per horizon', status='operational', priority='A'),
        EnumRegistryEntry(id='per_target_horizon', description='separate artifacts per target-horizon pair', status='operational', priority='A'),
        EnumRegistryEntry(id='flat', description='flat artifact directory', status='operational', priority='A'),
    ),
    compatible_with={},
    incompatible_with={},
)
