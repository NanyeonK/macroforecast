from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='additional_preprocessing',
    layer='2_preprocessing',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='none',
            description='none',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='smoothing_ma',
            description='smoothing moving average',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='ema',
            description='ema',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='hp_filter',
            description='hp filter',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='bandpass_filter',
            description='bandpass filter',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
