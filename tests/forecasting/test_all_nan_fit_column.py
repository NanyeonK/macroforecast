"""A feature column that is entirely NaN over the fit window (e.g. a late-starting
FRED-MD series like ACOGNO, absent over an early expanding-window fit sample) must
NOT empty the whole fit sample.

Regression for the raw-wide-predictor zero-rows bug: the training slice uses a
row-wise ``dropna``; a single all-NaN feature column made EVERY row NaN, so the fit
sample emptied and the arm silently produced ZERO forecasts even though every other
predictor was dense. ``_drop_all_nan_fit_columns`` drops such uninformative columns
(judged only on the fit window, hence leak-free) before slicing.
"""
import numpy as np
import pandas as pd

import macroforecast as mf


def _bundle_with_late_series(n=120, late_start=80):
    idx = pd.date_range("1990-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(0)
    panel = pd.DataFrame({f"s{i}": rng.normal(size=n) for i in range(4)}, index=idx)
    late = np.full(n, np.nan)
    late[late_start:] = rng.normal(size=n - late_start)  # NaN before late_start
    panel["late"] = late
    panel["Y"] = 0.5 * panel["s0"] + rng.normal(size=n)
    panel.index.name = "date"
    return mf.data.custom_dataset(panel, transform_codes={c: 1 for c in panel.columns})


def test_all_nan_fit_column_does_not_empty_fit_sample():
    # Test window (early 1993) precedes ``late``'s first observation (~1996-09), so the
    # raw ``late_lag0``/``late_lag1`` feature columns are all-NaN over every fit window.
    win = mf.window.from_cutoffs(
        test_start="1993-01-01", test_end="1993-06-01", mode="expanding",
        retrain_every=1,
    )
    feats = mf.feature_spec(
        target="Y", predictors=["s0", "s1", "s2", "s3", "late"],
        lags=(0, 1), target_lags=(1, 2, 3),
    )
    report = mf.forecasting.run(
        _bundle_with_late_series(), "ridge", window=win, features=feats,
        target="Y", horizons=[1], forecast_policy="direct",
    )
    fc = report.to_frame().dropna(subset=["prediction"])
    # Without the fix this is EMPTY (the all-NaN ``late`` columns nuke every row).
    assert not fc.empty, "all-NaN fit-window feature column emptied the fit sample"


def test_all_nan_column_prune_is_noop_when_all_dense():
    # No all-NaN column -> behaviour unchanged (forecasts still produced).
    win = mf.window.from_cutoffs(
        test_start="1993-01-01", test_end="1993-06-01", mode="expanding",
        retrain_every=1,
    )
    feats = mf.feature_spec(
        target="Y", predictors=["s0", "s1", "s2", "s3"], lags=(0, 1),
        target_lags=(1, 2, 3),
    )
    report = mf.forecasting.run(
        _bundle_with_late_series(), "ridge", window=win, features=feats,
        target="Y", horizons=[1], forecast_policy="direct",
    )
    assert not report.to_frame().dropna(subset=["prediction"]).empty
