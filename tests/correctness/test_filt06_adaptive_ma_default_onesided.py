"""Regression test for FILT-06.

adaptive_ma_rf_features() must default to a real-time-safe (one-sided) smoother,
matching the safe default of the underlying albama() callable. A two-sided
default produces look-ahead-biased features silently.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import macroforecast as mf


def test_adaptive_ma_rf_features_default_is_one_sided():
    panel = pd.DataFrame(
        {"x": np.r_[np.ones(8), np.ones(8) * 4.0]},
        index=pd.date_range("2000-01-31", periods=16, freq="ME"),
    )
    default = mf.feature_engineering.adaptive_ma_rf_features(
        panel, columns=["x"], n_estimators=6, min_samples_leaf=2,
        sample_fraction=0.8, warn_full_sample=False,
    )
    weight = default.attrs["macroforecast_feature_weight_results"]["x"]
    assert weight.mode == "one_sided"

    explicit_one = mf.feature_engineering.adaptive_ma_rf_features(
        panel, columns=["x"], n_estimators=6, min_samples_leaf=2,
        sample_fraction=0.8, sided="one", warn_full_sample=False,
    )
    pd.testing.assert_frame_equal(default, explicit_one)
