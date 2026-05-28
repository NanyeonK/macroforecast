"""Public feature-selection namespace.

This module re-exports the sklearn-style L3 selector classes from
``macroforecast.features.selection`` so the promoted
``macroforecast.feature_selection`` import path stays importable.
"""
from __future__ import annotations

from .features.selection import (
    Boruta,
    GeneticSelection,
    LassoPathSelector,
    RFE,
    StabilitySelection,
)

__all__ = [
    "Boruta",
    "RFE",
    "LassoPathSelector",
    "StabilitySelection",
    "GeneticSelection",
]
