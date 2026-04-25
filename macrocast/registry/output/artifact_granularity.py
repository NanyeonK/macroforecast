from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name='artifact_granularity',
    layer='5_output_provenance',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='per_target', description='one artifact set per target', status="registry_only", priority='B'),
        EnumRegistryEntry(id='per_target_horizon', description='separate artifacts per target-horizon pair', status="future", priority='B'),
        EnumRegistryEntry(id='aggregated', description='single aggregated artifact for all', status='operational', priority='A'),
        EnumRegistryEntry(id='hierarchical', description='hierarchical artifact tree', status="future", priority='B'),
    ),
    compatible_with={},
    incompatible_with={},
)
