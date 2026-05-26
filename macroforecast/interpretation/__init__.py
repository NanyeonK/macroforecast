"""Backward-compat alias for macroforecast.interpretation.

The canonical home for GIRF and LSTMHiddenState is now
macroforecast.layers.l7_interpretation.methods.
This module re-exports both classes for existing code that imports from
macroforecast.interpretation.

Original public namespace location (v0.9.5).
Phase 3g-bis -- body moved to layers/l7_interpretation/methods.py.
"""
from __future__ import annotations

from macroforecast.layers.l7_interpretation.methods import (  # noqa: F401
    GIRF,
    LSTMHiddenState,
)

__all__ = [
    "GIRF",
    "LSTMHiddenState",
]
