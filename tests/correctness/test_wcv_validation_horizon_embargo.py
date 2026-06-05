"""Regression test for WCV-2 / WCV-3.

from_cutoffs(..., horizon=H) must propagate a horizon-aware embargo to EVERY
validation method, not just 'expanding'. For an h-step target the validation
train/val boundary must be embargoed by at least horizon-1 so no training label
realizes inside the validation block. Previously last_block / poos /
rolling_blocks / blocked_kfold received embargo=None (effective 0).
"""
from __future__ import annotations

import pandas as pd

from macroforecast.window.core import from_cutoffs


def test_from_cutoffs_propagates_horizon_embargo_to_all_val_methods():
    for method in ("blocked_kfold", "last_block", "poos", "rolling_blocks", "expanding"):
        ws = from_cutoffs(
            estimation_start="1960-01",
            test_start="1980-01",
            test_end="1985-01",
            mode="expanding",
            val_method=method,
            horizon=12,
        )
        assert ws.val.embargo == 11, (method, ws.val.embargo)


def test_explicit_val_embargo_overrides_horizon_default():
    ws = from_cutoffs(
        estimation_start="1960-01", test_start="1980-01", test_end="1985-01",
        val_method="blocked_kfold", horizon=12, val_embargo=3,
    )
    assert ws.val.embargo == 3
