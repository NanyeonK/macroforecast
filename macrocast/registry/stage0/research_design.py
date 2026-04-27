from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='research_design',
    layer='0_meta',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='single_forecast_run',
            description='single resolved forecasting run',
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
            id='study_bundle',
            description='study bundle routed to a wrapper/orchestrator',
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id='replication_recipe',
            description='replication recipe study',
            status="operational",
            priority="A",
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
