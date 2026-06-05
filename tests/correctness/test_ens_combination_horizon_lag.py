"""Regression test for ENS-01 / ENS-02.

Data-driven forecast combination weights (inverse-MSPE and best-n) must use only
errors observable at the forecast origin. For an h-step forecast the most recent
usable error is for a target date <= origin = target_date - h. Perturbing the
forecast error at a target date that is NOT yet observable at a later row's
origin must not change that later combined forecast under horizon=h, while it
does change it under horizon=1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from macroforecast.forecasting.combination import combine_inverse_mspe, combine_best_n


def _frame_and_target(seed: int = 0):
    rng = np.random.default_rng(seed)
    n = 14
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    target = pd.Series(rng.normal(size=n), index=idx, name="y")
    frame = pd.DataFrame(
        {"A": target + rng.normal(scale=0.5, size=n),
         "B": target + rng.normal(scale=0.5, size=n)},
        index=idx,
    )
    return frame, target


def test_inverse_mspe_horizon3_ignores_unobservable_recent_error():
    frame, target = _frame_and_target()
    row = 10
    perturb_row = row - 1  # observable at origin for h=1 but NOT for h=3

    base_h3 = combine_inverse_mspe(frame, target, horizon=3)
    base_h1 = combine_inverse_mspe(frame, target, horizon=1)

    pert = frame.copy()
    pert.iloc[perturb_row, 0] += 50.0  # blow up model A's error at that row
    p_h3 = combine_inverse_mspe(pert, target, horizon=3)
    p_h1 = combine_inverse_mspe(pert, target, horizon=1)

    # h=3: row-(row-1) error not observable at row's origin -> weights unchanged.
    assert np.isclose(base_h3.iloc[row], p_h3.iloc[row])
    # h=1: that error IS observable -> the combined value changes.
    assert not np.isclose(base_h1.iloc[row], p_h1.iloc[row])


def test_best_n_horizon3_ignores_unobservable_recent_error():
    frame, target = _frame_and_target(seed=1)
    row = 10
    base_h3 = combine_best_n(frame, target, n=1, horizon=3)
    base_h1 = combine_best_n(frame, target, n=1, horizon=1)
    pert = frame.copy()
    pert.iloc[row - 1, 0] += 50.0
    p_h3 = combine_best_n(pert, target, n=1, horizon=3)
    p_h1 = combine_best_n(pert, target, n=1, horizon=1)
    assert np.isclose(base_h3.iloc[row], p_h3.iloc[row])
    assert not np.isclose(base_h1.iloc[row], p_h1.iloc[row])
