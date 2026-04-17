from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='data_domain',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='macro',
            description='macro',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='macro_finance',
            description='macro finance',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='housing',
            description='housing',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='energy',
            description='energy',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='labor',
            description='labor',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='regional',
            description='regional',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='panel_macro',
            description='panel macro',
            status='future',
            priority='B',
        ),
        EnumRegistryEntry(
            id='text_macro',
            description='text macro',
            status='future',
            priority='B',
        ),
        EnumRegistryEntry(
            id='mixed_domain',
            description='mixed domain',
            status='future',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
