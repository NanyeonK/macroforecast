from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='research_design',
    layer='0_meta',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='single_path_benchmark',
            description='single path benchmark study',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='controlled_variation',
            description='controlled variation study',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='orchestrated_bundle',
            description='orchestrated bundle study',
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id='replication_override',
            description='replication override study',
            status="operational",
            priority="A",
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
