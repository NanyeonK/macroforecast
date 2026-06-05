"""Regression test for the freq_align quarterly->monthly causal default.

The default rule for aligning a quarterly series onto a monthly grid was
'step_backward' (bfill), which pushes a quarter's value BACKWARD into the
earlier months of that quarter. A quarterly figure (e.g. Q1 GDP) is not released
until the quarter ends, so back-filling it onto January/February injects future
information -> look-ahead bias. The causal default 'step_forward' (ffill) carries
the last released quarterly value forward only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from macroforecast.preprocessing.clean import freq_align_quarterly_to_monthly_clean


def test_default_alignment_is_causal_no_backfill():
    idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    # Quarterly value observed at quarter-ends (Mar, Jun); NaN elsewhere.
    panel = pd.DataFrame({"Q": [np.nan, np.nan, 10.0, np.nan, np.nan, 20.0]}, index=idx)

    aligned = freq_align_quarterly_to_monthly_clean(panel, ["Q"])  # default rule

    # Jan, Feb have no released quarterly value yet -> must stay NaN (no backfill
    # of the March value). Old leaky default would fill them with 10.0.
    assert np.isnan(aligned["Q"].iloc[0])
    assert np.isnan(aligned["Q"].iloc[1])
    # March onward carries the released value forward.
    assert aligned["Q"].iloc[2] == 10.0
    assert aligned["Q"].iloc[3] == 10.0  # ffill April from Q1
    assert aligned["Q"].iloc[5] == 20.0
