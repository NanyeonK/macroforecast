from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def _series() -> pd.Series:
    return pd.Series(
        np.linspace(1.0, 12.0, 12),
        index=pd.date_range("2000-01-31", periods=12, freq="ME", name="date"),
        name="x",
    )


def test_hp_filter_returns_filter_result_and_matches_feature_wrapper() -> None:
    series = _series()

    direct = mf.filters.hp_filter(series, lamb=1600.0)
    features = mf.feature_engineering.hp_filter_features(
        pd.DataFrame({"x": series}),
        columns=["x"],
        lamb=1600.0,
        component="both",
        warn_full_sample=False,
    )

    assert isinstance(direct, mf.filters.FilterResult)
    assert list(direct.values.columns) == ["cycle", "trend"]
    assert direct.metadata["method"] == "hp_filter"
    assert features["x_hp_cycle"].to_numpy() == pytest.approx(
        direct.values["cycle"].to_numpy()
    )
    assert features["x_hp_trend"].to_numpy() == pytest.approx(
        direct.values["trend"].to_numpy()
    )


def test_hamilton_filter_returns_labeled_components() -> None:
    series = pd.Series(np.arange(1.0, 40.0), name="x")

    direct = mf.filters.hamilton_filter(
        series,
        h=2,
        p=2,
        fit_policy="full_sample",
        min_train_size=3,
    )

    assert list(direct.values.columns) == ["cycle", "trend"]
    assert direct.metadata["label_alignment"] == "components are labeled at t+h"
    assert direct.values["cycle"].notna().sum() > 0
    assert direct.values["trend"].notna().sum() > 0


def test_savitzky_golay_and_wavelet_filter_are_direct_callables() -> None:
    series = _series()

    smooth = mf.filters.savitzky_golay(series, window_length=5, polyorder=2)
    wavelet = mf.filters.wavelet_filter(series, n_levels=2)

    assert list(smooth.values.columns) == ["savgol"]
    assert smooth.metadata["backend"] == "scipy.signal.savgol_filter"
    assert list(wavelet.values.columns) == ["wA1", "wD1", "wA2", "wD2"]
    assert wavelet.metadata["fit_policy"] == "causal_rolling"


def test_filter_namespace_is_module_only_at_top_level() -> None:
    assert mf.filters.hp_filter is not None
    assert mf.filters.albama is not None
    assert not hasattr(mf, "hp_filter")
    assert not hasattr(mf.feature_engineering, "albama")
