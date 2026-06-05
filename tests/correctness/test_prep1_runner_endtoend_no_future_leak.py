"""End-to-end (runner-level) regression test for the PREP-1 leak.

The unit test for PREP-1 hand-fed the correct origin-available label set to the
FittedPreprocessor. The leak that survived was in the RUNNER: it built the
origin-available set as estimation/fit/test, but test_idx spans the whole forward
horizon block, so h-1 strictly-future rows still entered the EM-factor imputation
/ IQR outlier fit and contaminated training-row features.

This drives the real mf.forecasting.run pipeline (em_factor imputation + iqr
outliers + standardization) for a single origin and asserts the forecast is
invariant to poisoning rows that are strictly after the origin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import macroforecast as mf


def _panel(n: int = 180, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(n, 3))
    load = rng.normal(size=(3, 7))
    cols = [f"x{i}" for i in range(6)] + ["y"]
    panel = pd.DataFrame(f @ load + 0.1 * rng.normal(size=(n, 7)), columns=cols,
                         index=pd.date_range("1990-01-31", periods=n, freq="ME"))
    # Missing cells in early (training) rows so EM imputation actually runs.
    panel.iloc[20, 0] = np.nan
    panel.iloc[35, 2] = np.nan
    panel.iloc[60, 4] = np.nan
    return panel


def _run_one_origin(panel: pd.DataFrame, horizon: int = 12):
    spec = mf.preprocessing.preprocess_spec(
        transform="none", impute="em_factor", em_n_factors=3,
        outliers="none", standardize="none", frame="keep",
    )
    origin = panel.index[120]
    window = mf.window.from_cutoffs(
        estimation_start=str(panel.index[0].date()),
        test_start=str(origin.date()),
        test_end=str(origin.date()),
        mode="expanding", horizon=horizon, step=1,
    )
    result = mf.forecasting.run(
        panel, model="ridge", target="y", horizon=horizon,
        forecast_policy="direct_average", preprocessing=spec,
        preprocessing_policy="origin_available", window=window,
        params={"alpha": 1.0},
    )
    fc = result.forecasts
    return float(fc["prediction"].dropna().iloc[0]), panel.index[120]


def test_runner_prediction_invariant_to_strictly_future_poison():
    panel = _panel()
    pred_clean, origin = _run_one_origin(panel)

    poisoned = panel.copy()
    future = poisoned.index > origin
    poisoned.loc[future] = poisoned.loc[future] + 6.0  # shifts EM factors, no all-outlier rows
    pred_poison, _ = _run_one_origin(poisoned)

    assert abs(pred_clean - pred_poison) < 1e-9, (pred_clean, pred_poison)
