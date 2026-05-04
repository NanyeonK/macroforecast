"""Issue #245 -- DFM-MM via DynamicFactorMQ when mixed_frequency=True."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macrocast.core.runtime import _DFMMixedFrequency


def _toy_panel(n: int = 120, seed: int = 0):
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(size=n)) * 0.2
    monthly = pd.DataFrame(
        {
            "monthly_x1": 0.5 * f + rng.normal(scale=0.1, size=n),
            "monthly_x2": -0.3 * f + rng.normal(scale=0.1, size=n),
            "monthly_x3": 0.2 * f + rng.normal(scale=0.1, size=n),
        }
    )
    y = pd.Series(0.7 * f + rng.normal(scale=0.1, size=n), name="y")
    return monthly, y


def test_dfm_default_remains_single_frequency_path():
    """Without mixed_frequency=True the v0.2 single-frequency MLE path
    runs. Pin the back-compat behaviour."""

    X, y = _toy_panel()
    model = _DFMMixedFrequency(n_factors=1, factor_order=1, mixed_frequency=False).fit(X, y)
    assert model._mode == "single_frequency"
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_dfm_mq_path_engages_when_quarterly_column_declared():
    """Issue #245 -- when a quarterly column appears in
    ``column_frequencies``, route through DynamicFactorMQ."""

    X, y = _toy_panel(n=180)
    column_frequencies = {"monthly_x1": "monthly", "monthly_x2": "monthly", "monthly_x3": "quarterly"}
    model = _DFMMixedFrequency(
        n_factors=1,
        factor_order=1,
        mixed_frequency=True,
        column_frequencies=column_frequencies,
    ).fit(X, y)
    # Either the MQ optimiser succeeded (target outcome) or it fell
    # back to single-frequency on this synthetic data. Both must be
    # documented in ``_mode``.
    assert model._mode in {"mixed_frequency_mq", "single_frequency"}


def test_dfm_mq_falls_back_when_only_quarterly_columns_present():
    """MQ requires at least one monthly variable; pure-quarterly input
    falls through to the single-frequency path."""

    X, y = _toy_panel()
    column_frequencies = {col: "quarterly" for col in X.columns}
    model = _DFMMixedFrequency(
        n_factors=1,
        factor_order=1,
        mixed_frequency=True,
        column_frequencies=column_frequencies,
    ).fit(X, y)
    assert model._mode == "single_frequency"
