"""ADD: MASE + seasonal-naive scale + ACF1 (forecast::accuracy parity)."""
from __future__ import annotations
import numpy as np, pandas as pd
import macroforecast as mf


def test_mase_matches_definition():
    y_train = pd.Series(np.arange(1, 25, dtype=float))            # naive(m=1) MAE = 1
    y_true = pd.Series([25.0, 26.0, 27.0])
    y_pred = pd.Series([25.5, 25.0, 27.5])                        # |err| = .5,1,.5
    assert mf.metrics.seasonal_naive_mae(y_train, m=1) == 1.0
    assert mf.metrics.mase(y_true, y_pred, y_train, m=1) == (0.5 + 1.0 + 0.5) / 3
    # seasonal scaling uses the m-step naive denominator
    assert mf.metrics.seasonal_naive_mae(y_train, m=12) == 12.0
    assert mf.metrics.mase(y_true, y_pred, y_train, m=12) == ((0.5 + 1.0 + 0.5) / 3) / 12.0


def test_mase_registered_and_error_metric():
    assert mf.metrics.get_metric("mase") is mf.metrics.mase
    # lower-is-better -> ascending ranking (not in higher-is-better set)
    from macroforecast.metrics import _metric_ascending
    assert _metric_ascending("mase") is True


def test_acf1_canonical():
    assert mf.metrics.acf1([1, 2, 3, 4]) == 0.25  # gamma_1/gamma_0 = 1.25/5.0
    assert np.isnan(mf.metrics.acf1([5.0]))
