"""Re-create evaluation_scale as a Layer 2 preprocessing axis (mirror of the
PreprocessContract field). Not part of 1.5 — lives in 2_preprocessing because
it is a preprocess-contract field with a direct runtime effect at the
preprocessing boundary.
"""
from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='evaluation_scale',
    layer='2_preprocessing',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='original_scale',
            description='evaluate metrics on the original (untransformed) scale',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='raw_level',
            description='alias of original_scale; legacy recipes use this label',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='transformed_scale',
            description='evaluate primary metrics on the transformed target scale and retain original-scale artifacts when available',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='both',
            description='report original-scale primary metrics plus transformed-scale metric artifacts',
            status='operational',
            priority='A',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
