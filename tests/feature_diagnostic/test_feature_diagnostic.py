from __future__ import annotations

import numpy as np
import pandas as pd

import macroforecast as mf


def _panel() -> pd.DataFrame:
    return mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=10, freq="MS"),
                "target": np.arange(10, dtype=float),
                "x1": np.arange(10, dtype=float),
                "x2": np.arange(10, dtype=float) * 2.0,
                "x3": [1.0, np.nan, 1.5, 1.6, 1.8, 2.0, 2.3, 2.4, 2.6, 2.8],
            }
        ),
        date="date",
        metadata={"dataset": "custom", "frequency": "monthly"},
    )


def test_diagnose_features_accepts_feature_set_and_attaches_metadata() -> None:
    features = mf.feature_engineering.feature_spec(
        target="target",
        horizon=1,
        predictors=("x1", "x2", "x3"),
        lags=(0, 1),
        rolling_windows=(2,),
        pca_components=1,
    ).fit_transform(_panel())

    report = mf.diagnose_features(
        features,
        include_correlation=True,
        selections={"origin_1": [features.X.columns[0]], "origin_2": list(features.X.columns[:2])},
    )

    assert report.overview["n_features"] == features.X.shape[1]
    assert report.overview["feature_metadata_available"] is True
    assert report.correlation is not None
    assert report.lags is not None
    assert "feature_diagnostic" in report.metadata
    assert report.selection_stability is not None
    assert report.selection_stability.iloc[0]["selected_count"] == 2
    assert report.lags.attrs["macroforecast_metadata"] == report.metadata


def test_feature_correlation_returns_sorted_pairs_with_metadata() -> None:
    lagged = mf.feature_engineering.lag(_panel(), columns=["x1", "x2"], lags=(0,))

    pairs = mf.feature_diagnostic.feature_correlation(lagged, threshold=0.99)

    assert list(pairs.loc[0, ["feature_a", "feature_b"]]) == ["x1_lag0", "x2_lag0"]
    assert pairs.loc[0, "abs_correlation"] == 1.0
    assert pairs.loc[0, "operation_a"] == "lag"
    assert pairs.loc[0, "source_b"] == "x2"


def test_factor_diagnostics_detects_pca_components() -> None:
    factors = mf.feature_engineering.pca_features(
        _panel(),
        columns=["x1", "x2", "x3"],
        n_components=2,
        fit_policy="full_sample",
        min_train_size=3,
        warn_full_sample=False,
    )

    diagnostics = mf.feature_diagnostic.factor_diagnostics(factors)

    assert list(diagnostics["feature"]) == ["pc1", "pc2"]
    assert set(diagnostics["operation"]) == {"pca"}
    assert diagnostics["variance_share"].sum() == 1.0


def test_lag_and_marx_diagnostics_parse_feature_names() -> None:
    marx = mf.feature_engineering.moving_average_ladder(
        _panel(),
        columns=["x1"],
        windows=(1, 2, 4),
        shift=1,
    )

    lags = mf.feature_diagnostic.lag_diagnostics(marx)
    marx_rows = mf.feature_diagnostic.marx_diagnostics(marx)

    assert set(lags["window"]) == {1, 2, 4}
    assert set(marx_rows["window"]) == {1, 2, 4}
    assert set(marx_rows["lag"]) == {1}
    assert marx_rows.loc[marx_rows["window"] == 4, "marx_formula"].iloc[0] == "mean(x1[t-1]...x1[t-4])"


def test_compare_feature_stages_reports_column_deltas() -> None:
    base = _panel()[["x1", "x2"]]
    lagged = mf.feature_engineering.lag(_panel(), columns=["x1", "x2"], lags=(0, 1))

    comparison = mf.feature_diagnostic.compare_feature_stages(
        {"base": base, "lagged": lagged}
    )

    assert comparison.loc["base", "n_features"] == 2
    assert comparison.loc["lagged", "added_from_previous"] == 4
    assert comparison.loc["lagged", "removed_from_previous"] == 2


def test_selection_stability_accepts_indicator_frame() -> None:
    selections = pd.DataFrame(
        {
            "x1": [True, True, False],
            "x2": [False, True, False],
            "x3": [False, False, False],
        },
        index=["w1", "w2", "w3"],
    )

    stability = mf.feature_diagnostic.selection_stability(selections)

    assert stability.loc["x1", "selection_rate"] == 2 / 3
    assert stability.loc["x2", "selected_count"] == 1
    assert stability.loc["x3", "selected_count"] == 0
