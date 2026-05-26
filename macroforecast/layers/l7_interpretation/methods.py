"""Public L7 interpretation class wrappers.

Exposes two class-based wrappers for L7 interpretation operations that
operate on fitted model artifacts. Each class provides a ``compute(...)``
method that delegates numeric work to the existing private runtime helpers
in ``macroforecast.core.runtime``.

Classes
-------
- :class:`GIRF`             -- Pesaran-Shin (1998) Generalized IRF.
- :class:`LSTMHiddenState`  -- Karpathy (2015) LSTM hidden-state importance.

Usage::

    from macroforecast.layers.l7_interpretation.methods import GIRF, LSTMHiddenState

    # For a fitted VAR result (from macroforecast.functions.var_fit):
    girf = GIRF()
    importance_df = girf.compute(fitted_var, n_periods=12)

    # For a fitted LSTM result (from macroforecast.functions.lstm_fit):
    lstm_interp = LSTMHiddenState()
    importance_df = lstm_interp.compute(fitted_lstm, X)

Cycle 63 -- L7 interpretation class wrappers (GIRF + LSTMHiddenState).
Phase 3e -- collocated into layers/l7_interpretation/.
"""
from __future__ import annotations

# Re-export from the original public module for backward compat.
from macroforecast.interpretation import (  # noqa: F401
    GIRF,
    LSTMHiddenState,
)

__all__ = [
    "GIRF",
    "LSTMHiddenState",
]
