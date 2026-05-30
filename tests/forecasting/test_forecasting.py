from __future__ import annotations

import json

import numpy as np
import pandas as pd

import macroforecast as mf


def teardown_function() -> None:
    mf.meta.reset_config()


def _panel(n: int = 48) -> pd.DataFrame:
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    x = np.linspace(0.0, 1.0, n)
    return pd.DataFrame(
        {
            "y": 1.0 + 2.0 * x,
            "x1": x,
            "x2": np.sin(np.arange(n) / 3.0),
        },
        index=idx,
    )


def _window() -> mf.window.WindowSpec:
    return mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=24),
        val=mf.window.val_last_block(size=8),
        test=mf.window.test_origins(horizon=1, step=6),
    )


def test_preprocess_and_feature_specs_fit_transform() -> None:
    panel = _panel()
    pre = mf.preprocessing.preprocess_spec(
        transform="none",
        outliers="none",
        impute="mean",
        frame="keep",
    )
    fitted_pre = pre.fit(panel.iloc[:30])
    processed = fitted_pre.transform(panel.iloc[30:34], history=panel.iloc[:30])

    assert isinstance(processed, mf.preprocessing.PreprocessedData)
    assert list(processed.panel.index) == list(panel.index[30:34])

    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0, 1),
        pca_components=1,
    )
    fitted_features = features.fit(fitted_pre.processed_train)
    feature_set = fitted_features.transform(panel, index=panel.index[24:30])

    assert isinstance(feature_set, mf.feature_engineering.FeatureSet)
    assert "pc1" in feature_set.X.columns
    assert feature_set.y.shape[1] == 1


def test_forecasting_runner_connects_window_features_and_model() -> None:
    panel = _panel()
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0, 1),
        pca_components=1,
    )

    result = mf.forecasting.run(panel, "ols", window=_window(), features=features)
    table = result.to_frame()

    assert isinstance(result, mf.forecasting.ForecastResult)
    assert {"date", "origin", "prediction", "actual", "model"}.issubset(table.columns)
    assert set(table["model"]) == {"ols"}
    assert result.metadata["features"]["pca_components"] == 1
    json.dumps(result.to_dict())


def test_forecasting_runner_supports_multiple_models_and_stage_policies() -> None:
    panel = _panel()
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0, 1),
    )

    result = mf.forecasting.run(
        panel,
        ["ols", "ridge"],
        window=_window(),
        features=features,
        feature_policy=mf.window.stage_policy("fit_window"),
        params={"ridge": {"alpha": 0.1}},
    )
    table = result.to_frame()

    assert set(table["model"]) == {"ols", "ridge"}
    assert result.metadata["stage_policies"]["feature_engineering"]["scope"] == "fit_window"
    assert len(result.metadata["models"]) == 2
    ridge_selection = table.loc[table["model"] == "ridge", "selection"].dropna().iloc[0]
    assert ridge_selection["window"] == "explicit_splits"
    assert ridge_selection["metadata"]["split_source"] == "explicit"


def test_forecasting_runner_reads_meta_stage_defaults_and_metadata_level() -> None:
    mf.meta.configure(default_feature_scope="origin_available", metadata_level="minimal")
    panel = _panel()
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0, 1),
    )

    result = mf.forecasting.run(panel, "ols", window=_window(), features=features)

    assert result.metadata["stage_policies"]["feature_engineering"]["scope"] == "origin_available"
    assert result.metadata["run"]["config"]["metadata_level"] == "minimal"
    assert result.metadata["stages"] == []


def test_forecasting_runner_supports_window_local_preprocessing() -> None:
    panel = _panel()
    pre = mf.preprocessing.preprocess_spec(
        transform="none",
        outliers="none",
        impute="mean",
        standardize="zscore",
        standardize_columns="predictors",
        frame="keep",
    )
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0, 1),
        pca_components=1,
    )

    result = mf.forecasting.run(panel, "ols", window=_window(), preprocessing=pre, features=features)
    table = result.to_frame()

    assert not table.empty
    assert table["preprocessed"].all()
    assert result.metadata["preprocessing"]["options"]["standardize"] == "zscore"
    assert result.metadata["stage_policies"]["preprocessing"]["scope"] == "origin_available"

    fit_window_pre = mf.preprocessing.preprocess_spec(
        transform="none",
        outliers="none",
        impute="mean",
        standardize="zscore",
        standardize_columns="predictors",
        frame="keep",
    )
    fit_window_result = mf.forecasting.run(
        panel,
        "ols",
        window=_window(),
        preprocessing=fit_window_pre,
        preprocessing_policy=mf.window.stage_policy("fit_window"),
        features=features,
    )
    assert not fit_window_result.to_frame().empty
    assert fit_window_result.metadata["stage_policies"]["preprocessing"]["scope"] == "fit_window"


def test_forecast_combination_methods() -> None:
    idx = pd.date_range("2000-01-31", periods=4, freq="ME", name="date")
    forecasts = pd.DataFrame(
        {
            "m1": [1.0, 2.0, 3.0, 4.0],
            "m2": [1.5, 2.5, 2.5, 4.5],
            "m3": [0.5, 1.5, 3.5, 3.5],
        },
        index=idx,
    )
    y = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)

    assert mf.forecasting.combine_mean(forecasts).name == "combined"
    assert mf.forecasting.combine_median(forecasts).iloc[0] == 1.0
    assert len(mf.forecasting.combine_trimmed_mean(forecasts, trim=0.1)) == 4
    assert len(mf.forecasting.combine_winsorized_mean(forecasts)) == 4
    assert len(mf.forecasting.combine_inverse_mspe(forecasts, y)) == 4
    assert len(mf.forecasting.combine_best_n(forecasts, y, n=2)) == 4
