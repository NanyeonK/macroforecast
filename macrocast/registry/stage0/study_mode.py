from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='study_mode',
    layer='0_meta',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='single_path_benchmark_study',
            description='single path benchmark study',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='controlled_variation_study',
            description='controlled variation study',
            status='registry_only',
            priority="A",
        ),
        EnumRegistryEntry(
            id='orchestrated_bundle_study',
            description='orchestrated bundle study',
            status='planned',
            priority="A",
        ),
        EnumRegistryEntry(
            id='replication_override_study',
            description='replication override study',
            status='planned',
            priority="A",
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
