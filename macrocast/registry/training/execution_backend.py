from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='execution_backend',
    layer='3_training',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='local_cpu',
            description='local cpu',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='local_gpu',
            description='local gpu',
            status='future',
            priority='B',
        ),
        EnumRegistryEntry(
            id='ray',
            description='ray',
            status='future',
            priority='B',
        ),
        EnumRegistryEntry(
            id='dask',
            description='dask',
            status='future',
            priority='B',
        ),
        EnumRegistryEntry(
            id='joblib',
            description='joblib',
            status="operational",
            priority='A',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
