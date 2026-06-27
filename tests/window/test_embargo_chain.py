"""val_splits_for_origin must honour the WindowSpec-level embargo.

Regression: split()/to_table() resolve the embargo as
val.embargo -> WindowSpec.embargo (if non-zero) -> estimation.embargo, but
val_splits_for_origin skipped the WindowSpec level and went straight to
estimation.embargo. With WindowSpec(embargo=X, estimation.embargo=0) the planned
splits then disagreed with the per-origin splits.
"""
import numpy as np
import pandas as pd

from macroforecast.window.core import EstimationWindow, ValWindow, WindowSpec


def _gap(split):
    train_idx, val_idx = split
    train = np.asarray(list(train_idx))
    val = np.asarray(list(val_idx))
    return int(val.min()) - int(train.max()) - 1


def _spec(embargo):
    # val.embargo=None and estimation.embargo=0 so the only embargo source is the
    # WindowSpec level; an explicit non-default val keeps __post_init__ from
    # copying WindowSpec.embargo into val.
    return WindowSpec(
        val=ValWindow(method="last_block", size=4, embargo=None),
        estimation=EstimationWindow(embargo=0),
        embargo=embargo,
    )


def test_per_origin_embargo_matches_split():
    spec = _spec(embargo=3)
    split_gaps = {_gap(s) for s in spec.split(50)}
    idx = pd.date_range("2000-01-01", periods=90, freq="MS")
    origin = idx[70]
    origin_gaps = {_gap(s) for s in spec.val_splits_for_origin(idx, origin)}
    assert split_gaps and origin_gaps
    assert split_gaps == {3}
    assert origin_gaps == {3}, f"per-origin embargo {origin_gaps} != WindowSpec embargo 3"


def test_zero_windowspec_embargo_falls_through_to_estimation():
    # WindowSpec.embargo == 0 must fall through to estimation.embargo (here 2).
    spec = WindowSpec(
        val=ValWindow(method="last_block", size=4, embargo=None),
        estimation=EstimationWindow(embargo=2),
        embargo=0,
    )
    idx = pd.date_range("2000-01-01", periods=90, freq="MS")
    origin = idx[70]
    assert {_gap(s) for s in spec.val_splits_for_origin(idx, origin)} == {2}
