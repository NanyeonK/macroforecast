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

    report = mf.feature_analysis.diagnose_features(
        features,
        include_correlation=True,
        include_correlation_matrix=True,
        include_target_correlation=True,
        target=features.y.iloc[:, 0],
        include_lag_autocorrelation=True,
        include_factor_timeseries=True,
        selection_similarity_metric="jaccard",
        selections={"origin_1": [features.X.columns[0]], "origin_2": list(features.X.columns[:2])},
    )

    assert report.overview["n_features"] == features.X.shape[1]
    assert report.overview["feature_metadata_available"] is True
    assert report.correlation is not None
    assert report.correlation_matrix is not None
    assert report.target_correlation is not None
    assert report.factor_timeseries is not None
    assert report.lags is not None
    assert report.lag_autocorrelation is not None
    assert "feature_analysis" in report.metadata
    assert report.selection_stability is not None
    assert report.selection_similarity is not None
    assert report.selection_stability.iloc[0]["selected_count"] == 2
    assert report.lags.attrs["macroforecast_metadata"] == report.metadata
    assert mf.feature_analysis.feature_overview is mf.feature_diagnostic.feature_overview


def test_custom_feature_diagnostic_wraps_user_callable() -> None:
    features = mf.feature_engineering.feature_spec(
        target="target",
        horizon=1,
        predictors=("x1", "x2", "x3"),
        lags=(0,),
    ).fit_transform(_panel())

    def missing_overview(X, *, feature_metadata=None, metadata=None, label="custom"):
        return pd.DataFrame(
            [
                {
                    "label": label,
                    "n_features": X.shape[1],
                    "metadata_keys": len(metadata or {}),
                    "has_feature_metadata": feature_metadata is not None,
                }
            ]
        )

    out = mf.feature_analysis.custom_feature_diagnostic(
        features,
        missing_overview,
        name="missing_overview",
        metadata={"owner": "test"},
        label="demo",
    )

    assert out.loc[0, "label"] == "demo"
    assert bool(out.loc[0, "has_feature_metadata"]) is True
    assert out.attrs["macroforecast_metadata_schema"]["kind"] == "custom_feature_diagnostic"
    assert out.attrs["macroforecast_metadata_schema"]["method"] == "missing_overview"
    assert "custom_feature_diagnostic" in out.attrs["macroforecast_metadata"]


def test_feature_correlation_returns_sorted_pairs_with_metadata() -> None:
    lagged = mf.feature_engineering.lag(_panel(), columns=["x1", "x2"], lags=(0,))
    metadata = lagged.attrs["macroforecast_feature_metadata"].copy()
    metadata.loc[metadata["source"] == "x1", "block"] = "real"
    metadata.loc[metadata["source"] == "x2", "block"] = "financial"

    pairs = mf.feature_analysis.feature_correlation(lagged, threshold=0.99, feature_metadata=metadata)
    cross_block = mf.feature_analysis.feature_correlation(
        lagged,
        threshold=0.99,
        feature_metadata=metadata,
        scope="cross_block",
    )

    assert list(pairs.loc[0, ["feature_a", "feature_b"]]) == ["x1_lag0", "x2_lag0"]
    assert pairs.loc[0, "abs_correlation"] == 1.0
    assert pairs.loc[0, "operation_a"] == "lag"
    assert pairs.loc[0, "source_b"] == "x2"
    assert set(cross_block.loc[0, ["block_a", "block_b"]]) == {"real", "financial"}

    target_corr = mf.feature_analysis.feature_target_correlation(lagged, _panel()["target"])

    assert {"feature", "target", "correlation", "n_obs"}.issubset(target_corr.columns)


def test_feature_correlation_matrix_supports_cluster_order() -> None:
    lagged = mf.feature_engineering.lag(_panel(), columns=["x1", "x2", "x3"], lags=(0,))

    matrix = mf.feature_analysis.feature_correlation_matrix(lagged, order="clustered")

    assert matrix.shape == (3, 3)
    assert matrix.attrs["macroforecast_metadata_schema"]["kind"] == "feature_correlation_matrix"
    assert matrix.attrs["macroforecast_metadata_schema"]["order"] == "clustered"


def test_factor_diagnostics_detects_pca_components() -> None:
    factors = mf.feature_engineering.pca_features(
        _panel(),
        columns=["x1", "x2", "x3"],
        n_components=2,
        fit_policy="full_sample",
        min_train_size=3,
        warn_full_sample=False,
    )

    diagnostics = mf.feature_analysis.factor_diagnostics(factors)

    assert list(diagnostics["feature"]) == ["pc1", "pc2"]
    assert set(diagnostics["operation"]) == {"pca"}
    assert diagnostics["variance_share"].sum() == 1.0

    variance = mf.feature_analysis.factor_variance(factors)
    loadings = mf.feature_analysis.factor_loadings(factors, source_data=_panel()[["x1", "x2", "x3"]])
    time_series = mf.feature_analysis.factor_timeseries(factors)

    assert list(variance["feature"]) == ["pc1", "pc2"]
    assert variance["cumulative_variance_share"].iloc[-1] == 1.0
    assert {"factor", "source", "loading", "abs_loading"}.issubset(loadings.columns)
    assert set(loadings["factor"]) == {"pc1", "pc2"}
    assert {"date", "factor", "value"}.issubset(time_series.columns)
    assert set(time_series["factor"]) == {"pc1", "pc2"}


def test_lag_and_marx_diagnostics_parse_feature_names() -> None:
    marx = mf.feature_engineering.moving_average_ladder(
        _panel(),
        columns=["x1"],
        windows=(1, 2, 4),
        shift=1,
    )

    lags = mf.feature_analysis.lag_diagnostics(marx)
    marx_rows = mf.feature_analysis.marx_diagnostics(marx)
    weight_decay = mf.feature_analysis.marx_weight_decay(marx)

    assert set(lags["window"]) == {1, 2, 4}
    assert set(marx_rows["window"]) == {1, 2, 4}
    assert set(marx_rows["lag"]) == {1}
    assert marx_rows.loc[marx_rows["window"] == 4, "marx_formula"].iloc[0] == "mean(x1[t-1]...x1[t-4])"

    acf = mf.feature_analysis.lag_autocorrelation(marx, max_lag=2)
    decay = mf.feature_analysis.lag_correlation_decay(marx)

    assert set(acf["lag"]) == {0, 1, 2}
    assert acf.attrs["macroforecast_metadata_schema"]["autocorrelation_kind"] == "acf"
    assert {"feature", "source", "correlation"}.issubset(decay.columns)
    assert set(weight_decay["window"]) == {1, 2, 4}
    assert np.allclose(weight_decay.groupby("feature")["weight"].sum().to_numpy(), 1.0)


def test_marx_diagnostics_respect_starting_lag() -> None:
    idx = pd.date_range("2021-01-31", periods=5, freq="ME", name="date")
    features = pd.DataFrame(
        {
            "x_ma4_lag2": np.arange(5, dtype=float),
        },
        index=idx,
    )

    marx = mf.feature_analysis.marx_diagnostics(features)
    weights = mf.feature_analysis.marx_weight_decay(features)

    assert marx.loc[0, "marx_formula"] == "mean(x[t-2]...x[t-5])"
    assert list(weights["lag"]) == [2, 3, 4, 5]
    assert np.allclose(weights["weight"], 0.25)


def test_compare_feature_stages_reports_column_deltas() -> None:
    base = _panel()[["x1", "x2"]]
    lagged = mf.feature_engineering.lag(_panel(), columns=["x1", "x2"], lags=(0, 1))

    comparison = mf.feature_analysis.compare_feature_stages(
        {"base": base, "lagged": lagged}
    )
    shift = mf.feature_analysis.stage_distribution_shift({"base": base, "lagged": lagged})

    assert comparison.loc["base", "n_features"] == 2
    assert comparison.loc["lagged", "added_from_previous"] == 4
    assert comparison.loc["lagged", "removed_from_previous"] == 2
    assert {"stage_a", "stage_b", "feature", "ks_statistic"}.issubset(shift.columns)


def test_selection_stability_accepts_indicator_frame() -> None:
    selections = pd.DataFrame(
        {
            "x1": [True, True, False],
            "x2": [False, True, False],
            "x3": [False, False, False],
        },
        index=["w1", "w2", "w3"],
    )

    stability = mf.feature_analysis.selection_stability(selections)

    assert stability.loc["x1", "selection_rate"] == 2 / 3
    assert stability.loc["x2", "selected_count"] == 1
    assert stability.loc["x3", "selected_count"] == 0

    similarity = mf.feature_analysis.selection_similarity(
        selections,
        metric="kuncheva",
        all_features=["x1", "x2", "x3"],
    )

    assert {"origin_a", "origin_b", "score", "overlap"}.issubset(similarity.columns)
    assert similarity.attrs["macroforecast_metadata_schema"]["metric"] == "kuncheva"


def test_kuncheva_similarity_requires_equal_selection_size() -> None:
    equal = mf.feature_analysis.selection_similarity(
        {"w1": ["x1", "x2"], "w2": ["x1", "x3"]},
        metric="kuncheva",
        all_features=["x1", "x2", "x3", "x4"],
    )
    unequal = mf.feature_analysis.selection_similarity(
        {"w1": ["x1"], "w2": ["x1", "x2"]},
        metric="kuncheva",
        all_features=["x1", "x2", "x3", "x4"],
    )

    assert equal.loc[0, "score"] == 0.0
    assert pd.isna(unequal.loc[0, "score"])
