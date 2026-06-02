from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def test_albama_two_sided_root_leaf_weights_match_terminal_membership() -> None:
    series = pd.Series([1.0, 2.0, 3.0, 4.0], name="cpi")

    result = mf.feature_engineering.albama(
        series,
        mode="two_sided",
        n_estimators=1,
        min_samples_leaf=99,
        sample_fraction=1.0,
        replace=False,
        random_state=42,
    )

    assert result.smoothed.tolist() == pytest.approx([2.5, 2.5, 2.5, 2.5])
    assert result.weights.to_numpy() == pytest.approx(np.full((4, 4), 0.25))
    assert result.weights.sum(axis=0).tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert mf.feature_analysis.effective_window(result.weights).tolist() == [4, 4, 4, 4]
    assert (
        result.metadata["r_reference"]
        == "AlbaMA/AMA_main.R ranger keep.inbag terminalNodes loop"
    )


def test_albama_one_sided_is_causal_and_normalized() -> None:
    series = pd.Series([1.0, 2.0, 3.0, 4.0], name="cpi")

    result = mf.feature_engineering.albama(
        series,
        mode="one_sided",
        n_estimators=1,
        min_samples_leaf=99,
        sample_fraction=1.0,
        replace=False,
        random_state=42,
    )

    assert result.smoothed.tolist() == pytest.approx([1.0, 1.5, 2.0, 2.5])
    assert result.weights.to_numpy() == pytest.approx(
        np.array(
            [
                [1.0, 0.5, 1.0 / 3.0, 0.25],
                [0.0, 0.5, 1.0 / 3.0, 0.25],
                [0.0, 0.0, 1.0 / 3.0, 0.25],
                [0.0, 0.0, 0.0, 0.25],
            ]
        )
    )
    assert np.tril(result.weights.to_numpy(), k=-1).sum() == pytest.approx(0.0)
    shares = mf.feature_analysis.recent_weight_share(result.weights, mode="one_sided")
    assert shares["future_weight"].max() == pytest.approx(0.0)


def test_albama_constant_series_stays_constant() -> None:
    series = pd.Series(np.ones(12) * 7.0, name="constant")

    result = mf.feature_engineering.AlbaMA(
        mode="one_sided",
        n_estimators=8,
        min_samples_leaf=2,
        random_state=0,
    ).fit_transform(series)

    assert result.smoothed.dropna().tolist() == pytest.approx([7.0] * 12)
    assert result.weights.sum(axis=0).replace(
        0.0, np.nan
    ).dropna().tolist() == pytest.approx([1.0] * 12)


def test_adaptive_ma_rf_features_reuses_albama_smoother() -> None:
    panel = pd.DataFrame(
        {"x": np.r_[np.ones(6), np.ones(6) * 4.0]},
        index=pd.date_range("2000-01-31", periods=12, freq="ME"),
    )

    features = mf.feature_engineering.adaptive_ma_rf_features(
        panel,
        columns=["x"],
        n_estimators=4,
        min_samples_leaf=2,
        sample_fraction=0.8,
        sided="one",
        warn_full_sample=False,
    )

    assert list(features.columns) == ["x_albama"]
    assert "x" in features.attrs["macroforecast_feature_weight_results"]
    weight_result = features.attrs["macroforecast_feature_weight_results"]["x"]
    assert weight_result.mode == "one_sided"
    meta = features.attrs["macroforecast_metadata"][
        "feature_engineering_adaptive_ma_rf"
    ]
    assert meta["implementation"] == "macroforecast.feature_engineering.albama"
    assert meta["sample_fraction"] == pytest.approx(0.8)
