from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='predictor_family',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='target_lags_only',
            description='target lags only',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='all_macro_vars',
            description='all macro vars',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='all_except_target',
            description='all except target',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='category_based',
            description='category based',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='factor_only',
            description='factor only',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='text_only',
            description='text only',
            status='future',
            priority='B',
        ),
        EnumRegistryEntry(
            id='mixed_blocks',
            description='mixed blocks',
            status='future',
            priority='B',
        ),
        EnumRegistryEntry(
            id='handpicked_set',
            description='handpicked set',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
