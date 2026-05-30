from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
    assert (
        result.metadata["stage_policies"]["feature_engineering"]["scope"]
        == "fit_window"
    )
    assert len(result.metadata["models"]) == 2
    ridge_selection = table.loc[table["model"] == "ridge", "selection"].dropna().iloc[0]
    assert ridge_selection["window"] == "explicit_splits"
    assert ridge_selection["metadata"]["split_source"] == "explicit"


def test_forecasting_runner_can_disable_model_owned_selection() -> None:
    panel = _panel()
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0, 1),
    )

    result = mf.forecasting.run(
        panel,
        "ridge",
        window=_window(),
        features=features,
        params={"ridge": {"alpha": 0.1}},
        selection={"ridge": None},
    )
    table = result.to_frame()

    assert table["selection"].isna().all()
    assert result.metadata["models"][0]["spec"]["params"]["alpha"] == 0.1


def test_forecasting_runner_supports_pls_model(tmp_path) -> None:
    panel = _panel()
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0,),
    )

    result = mf.forecasting.run(
        panel,
        "pls",
        window=_window(),
        features=features,
        selection={"pls": None},
        model_store=tmp_path / "trained_model",
    )
    table = result.to_frame()

    assert set(table["model"]) == {"pls"}
    assert table["prediction"].notna().all()
    fit_metadata = table["stored_model"].dropna().iloc[0]
    metadata = json.loads(
        Path(fit_metadata["metadata_path"]).read_text(encoding="utf-8")
    )
    assert metadata["fit"]["fit"]["metadata"]["resolved_n_components"] == 2


def test_forecasting_runner_saves_trained_models(tmp_path) -> None:
    panel = _panel()
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0, 1),
    )

    result = mf.forecasting.run(
        panel,
        "ridge",
        window=_window(),
        features=features,
        selection=mf.selection.grid({"alpha": [0.01, 0.1]}),
        model_store=tmp_path / "trained_model",
    )
    table = result.to_frame()
    stored = table["stored_model"].dropna().iloc[0]
    model_path = tmp_path / "trained_model" / "ridge" / Path(stored["model_path"]).name
    metadata_path = (
        tmp_path / "trained_model" / "ridge" / Path(stored["metadata_path"]).name
    )

    assert model_path.exists()
    assert metadata_path.exists()
    with model_path.open("rb") as handle:
        fit = pickle.load(handle)
    assert fit.model == "ridge"
    assert "residuals" in fit.diagnostics
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["params"]["alpha"] in {0.01, 0.1}
    assert metadata["fit"]["fit"]["diagnostics"]["metrics"]["n"] > 0
    assert result.metadata["run"]["save_models"] is True


def test_forecasting_runner_supports_scaled_pca_model() -> None:
    panel = _panel()
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0, 1),
    )

    result = mf.forecasting.run(
        panel,
        "scaled_pca",
        window=_window(),
        features=features,
        params={"scaled_pca": {"n_components": 1}},
        selection={"scaled_pca": None},
    )
    table = result.to_frame()

    assert set(table["model"]) == {"scaled_pca"}
    assert table["prediction"].notna().all()
    assert result.metadata["models"][0]["spec"]["params"]["n_components"] == 1


def test_forecasting_runner_supports_supervised_pca_model() -> None:
    panel = _panel()
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0, 1),
    )

    result = mf.forecasting.run(
        panel,
        "supervised_pca",
        window=_window(),
        features=features,
        params={"supervised_pca": {"n_components": 1, "n_selected": 2}},
        selection={"supervised_pca": None},
    )
    table = result.to_frame()

    assert set(table["model"]) == {"supervised_pca"}
    assert table["prediction"].notna().all()
    assert result.metadata["models"][0]["spec"]["params"]["n_components"] == 1
    assert result.metadata["models"][0]["spec"]["params"]["n_selected"] == 2


def test_forecasting_runner_supports_supervised_scaled_pca_model() -> None:
    panel = _panel()
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0, 1),
    )

    result = mf.forecasting.run(
        panel,
        "supervised_scaled_pca",
        window=_window(),
        features=features,
        params={"supervised_scaled_pca": {"n_components": 1, "n_selected": 2}},
        selection={"supervised_scaled_pca": None},
    )
    table = result.to_frame()

    assert set(table["model"]) == {"supervised_scaled_pca"}
    assert table["prediction"].notna().all()
    assert result.metadata["models"][0]["spec"]["params"]["n_components"] == 1


def test_forecasting_runner_supports_svr_model() -> None:
    panel = _panel()
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0, 1),
    )

    result = mf.forecasting.run(
        panel,
        "svr",
        window=_window(),
        features=features,
        params={"svr": {"C": 1.0, "epsilon": 0.01}},
        selection={"svr": None},
    )
    table = result.to_frame()

    assert set(table["model"]) == {"svr"}
    assert table["prediction"].notna().all()
    assert result.metadata["models"][0]["spec"]["params"]["C"] == 1.0


def test_forecasting_runner_supports_random_forest_model() -> None:
    panel = _panel()
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0, 1),
    )

    result = mf.forecasting.run(
        panel,
        "random_forest",
        window=_window(),
        features=features,
        params={
            "random_forest": {
                "n_estimators": 10,
                "max_depth": 3,
                "random_state": 0,
                "n_jobs": 1,
            }
        },
        selection={"random_forest": None},
    )
    table = result.to_frame()

    assert set(table["model"]) == {"random_forest"}
    assert table["prediction"].notna().all()
    assert result.metadata["models"][0]["spec"]["params"]["n_estimators"] == 10


def test_forecasting_runner_records_quantile_predictions() -> None:
    panel = _panel()
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0, 1),
    )

    result = mf.forecasting.run(
        panel,
        "quantile_regression_forest",
        window=_window(),
        features=features,
        params={
            "quantile_regression_forest": {
                "n_estimators": 10,
                "random_state": 0,
                "quantile_levels": (0.1, 0.5, 0.9),
            }
        },
        selection={"quantile_regression_forest": None},
    )
    table = result.to_frame()

    assert set(table["model"]) == {"quantile_regression_forest"}
    assert table["prediction"].notna().all()
    quantiles = table["quantile_predictions"].dropna().iloc[0]
    assert set(quantiles) == {"0.1", "0.5", "0.9"}
    assert all(np.isfinite(value) for value in quantiles.values())


def test_forecasting_runner_supports_timeseries_and_ensemble_models() -> None:
    panel = _panel()
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0, 1),
    )

    result = mf.forecasting.run(
        panel,
        ["ar", "bagging"],
        window=_window(),
        features=features,
        params={
            "ar": {"n_lag": 2},
            "bagging": {"base": "ridge", "n_estimators": 3, "random_state": 0},
        },
        selection={"ar": None, "bagging": None},
    )
    table = result.to_frame()

    assert set(table["model"]) == {"ar", "bagging"}
    assert table["prediction"].notna().all()


@pytest.mark.parametrize(
    ("module", "model", "params"),
    [
        ("xgboost", "xgboost", {"n_estimators": 3, "max_depth": 2, "random_state": 0}),
        (
            "lightgbm",
            "lightgbm",
            {"n_estimators": 3, "num_leaves": 7, "random_state": 0, "verbose": -1},
        ),
        (
            "catboost",
            "catboost",
            {"n_estimators": 3, "max_depth": 2, "random_state": 0, "verbose": False},
        ),
    ],
)
def test_forecasting_runner_supports_optional_tree_backends_when_installed(
    tmp_path,
    module: str,
    model: str,
    params: dict[str, object],
) -> None:
    pytest.importorskip(module)
    panel = _panel()
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0, 1),
    )

    result = mf.forecasting.run(
        panel,
        model,
        window=_window(),
        features=features,
        params={model: params},
        selection={model: None},
        model_store=tmp_path / "trained_model",
    )
    table = result.to_frame()

    assert set(table["model"]) == {model}
    assert table["prediction"].notna().all()
    assert table["stored_model"].dropna().iloc[0]["save_error"] is None


def test_forecasting_runner_supports_nn_model_when_torch_is_available() -> None:
    pytest.importorskip("torch")
    panel = _panel()
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0, 1),
    )

    result = mf.forecasting.run(
        panel,
        "nn",
        window=_window(),
        features=features,
        params={
            "nn": {
                "hidden_layer_sizes": (8,),
                "max_epochs": 1,
                "batch_size": 8,
                "random_state": 0,
                "device": "cpu",
            },
        },
        selection={"nn": None},
    )
    table = result.to_frame()

    assert set(table["model"]) == {"nn"}
    assert table["prediction"].notna().all()
    assert result.metadata["models"][0]["spec"]["params"]["hidden_layer_sizes"] == [8]


def test_forecasting_runner_accepts_feature_set_input() -> None:
    panel = _panel()
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0, 1),
        add_time=True,
        time_month=True,
    )
    feature_set = features.fit_transform(panel)

    result = mf.forecasting.run(
        feature_set,
        "ridge",
        window=_window(),
        params={"ridge": {"alpha": 0.1}},
        selection={"ridge": None},
    )
    table = result.to_frame()

    assert set(table["model"]) == {"ridge"}
    assert table["prediction"].notna().all()
    assert result.metadata["features"]["spec"]["target"] == "y"
    assert result.metadata["features"]["output"]["n_features"] == feature_set.X.shape[1]


def test_forecasting_runner_supports_macro_random_forest_with_reference_backend(
    monkeypatch,
) -> None:
    panel = _panel()
    calls = {}

    class FakeMRF:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def _ensemble_loop(self):
            return {"pred_ensemble": np.full(len(calls["oos_pos"]), 4.0)}

    monkeypatch.setattr(
        "macroforecast.models.tree.MacroRandomForestRegressor._import_external",
        staticmethod(lambda: FakeMRF),
    )
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0,),
    )

    result = mf.forecasting.run(
        panel,
        "macro_random_forest",
        window=_window(),
        features=features,
        params={
            "macro_random_forest": {
                "B": 2,
                "x_columns": ["x1_lag0"],
                "S_columns": ["x1_lag0", "x2_lag0"],
            }
        },
        selection={"macro_random_forest": None},
    )
    table = result.to_frame()

    assert set(table["model"]) == {"macro_random_forest"}
    assert set(table["prediction"]) == {4.0}
    assert calls["x_pos"].tolist() == [1]
    assert calls["S_pos"].tolist() == [1, 2]
    assert result.metadata["models"][0]["spec"]["params"]["B"] == 2


def test_forecasting_runner_records_calendar_step_and_retune_reuse() -> None:
    panel = _panel(60)
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0, 1),
    )
    window = mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=24, retrain_every=2),
        val=mf.window.val_last_block(size=8, retune_every=2),
        test=mf.window.test_origins(horizon=1, step="2ME"),
    )

    result = mf.forecasting.run(
        panel,
        "ridge",
        window=window,
        features=features,
        selection=mf.selection.grid({"alpha": [0.01, 0.1]}),
    )
    table = result.to_frame()

    assert not table.empty
    assert result.metadata["window"]["test"]["step"] == "2ME"
    assert result.metadata["window"]["estimation"]["retrain_every"] == 2
    assert {row["test_step"] for row in table["window"]} == {"2ME"}
    assert {bool(row["retrain"]) for row in table["window"]} == {False, True}
    assert {bool(row["retune"]) for row in table["window"]} == {False, True}
    assert {selection["retuned"] for selection in table["selection"]} == {False, True}


def test_forecasting_runner_applies_feature_update_never() -> None:
    panel = _panel(60)
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0, 1),
        pca_components=1,
    )

    result = mf.forecasting.run(
        panel,
        "ols",
        window=_window(),
        features=features,
        feature_policy=mf.window.stage_policy("fit_window", update="never"),
    )
    updates = _stage_updates(result, "feature_engineering")

    assert len(updates) > 1
    assert updates[0] is True
    assert not any(updates[1:])


def test_forecasting_runner_applies_stage_update_on_retrain() -> None:
    panel = _panel(60)
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0, 1),
    )
    window = mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=24, retrain_every=2),
        val=mf.window.val_last_block(size=8),
        test=mf.window.test_origins(horizon=1, step=6),
    )

    result = mf.forecasting.run(
        panel,
        "ols",
        window=window,
        features=features,
        feature_policy=mf.window.stage_policy("fit_window", update="on_retrain"),
    )
    expected = [
        bool(item["row"]["retrain"]) for item in window.iter_origins(panel.index)
    ]

    assert _stage_updates(result, "feature_engineering") == expected


def test_forecasting_runner_applies_preprocessing_update_never() -> None:
    panel = _panel(60)
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
    )

    result = mf.forecasting.run(
        panel,
        "ols",
        window=_window(),
        preprocessing=pre,
        preprocessing_policy=mf.window.stage_policy("fit_window", update="never"),
        features=features,
    )
    updates = _stage_updates(result, "preprocessing")

    assert not result.to_frame().empty
    assert len(updates) > 1
    assert updates[0] is True
    assert not any(updates[1:])


def test_forecasting_runner_applies_date_offset_stage_update() -> None:
    panel = _panel(72)
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0, 1),
    )
    window = mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=24),
        val=mf.window.val_last_block(size=8),
        test=mf.window.test_origins(horizon=1, step="6ME"),
    )

    result = mf.forecasting.run(
        panel,
        "ols",
        window=window,
        features=features,
        feature_policy=mf.window.stage_policy("fit_window", update="12ME"),
    )
    updates = _stage_updates(result, "feature_engineering")

    assert len(updates) >= 3
    assert updates[:3] == [True, False, True]
    assert result.metadata["stage_policies"]["feature_engineering"]["update"] == "12ME"


def test_forecasting_runner_passes_target_and_exog_to_volatility_spec() -> None:
    panel = _panel()
    captured = {}

    class FakeVolFit:
        def __init__(self, target, X=None, scale=1.0):
            captured["target_name"] = target.name
            captured["exog_columns"] = tuple(X.columns)
            self.scale = scale

        def predict(self, X):
            return np.full(len(X), self.scale)

        def predict_variance(self, horizon=1):
            return pd.Series(np.full(int(horizon), self.scale + 1.0))

    def fake_volatility(y, *, X=None, scale=1.0):
        return FakeVolFit(y, X=X, scale=scale)

    spec = mf.models.ModelSpec(
        name="fake_volatility",
        family="volatility",
        fit_func=fake_volatility,
        input_kind="volatility",
        params={"scale": 3.0},
    )
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0,),
    )

    result = mf.forecasting.run(panel, spec, window=_window(), features=features)
    table = result.to_frame()

    assert set(table["prediction"]) == {3.0}
    assert set(table["variance_prediction"]) == {4.0}
    assert captured["target_name"] == "y_level_h1"
    assert captured["exog_columns"] == ("x1_lag0", "x2_lag0")


def test_forecasting_runner_reads_meta_stage_defaults_and_metadata_level() -> None:
    mf.meta.configure(
        default_feature_scope="origin_available", metadata_level="minimal"
    )
    panel = _panel()
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0, 1),
    )

    result = mf.forecasting.run(panel, "ols", window=_window(), features=features)

    assert (
        result.metadata["stage_policies"]["feature_engineering"]["scope"]
        == "origin_available"
    )
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

    result = mf.forecasting.run(
        panel, "ols", window=_window(), preprocessing=pre, features=features
    )
    table = result.to_frame()

    assert not table.empty
    assert table["preprocessed"].all()
    assert result.metadata["preprocessing"]["options"]["standardize"] == "zscore"
    assert (
        result.metadata["stage_policies"]["preprocessing"]["scope"]
        == "origin_available"
    )

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
    assert (
        fit_window_result.metadata["stage_policies"]["preprocessing"]["scope"]
        == "fit_window"
    )


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


def _stage_updates(result: mf.forecasting.ForecastResult, stage: str) -> list[bool]:
    return [
        bool(record["updated"])
        for record in result.metadata["stages"]
        if record["stage"] == stage
    ]
