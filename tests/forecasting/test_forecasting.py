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


def _mixed_panel(n: int = 48) -> mf.data.DataBundle:
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    t = np.arange(n, dtype=float)
    q_target = pd.Series(np.nan, index=idx, name="q_target")
    q_mask = idx.month.isin([3, 6, 9, 12])
    q_target.loc[q_mask] = 100.0 + 0.3 * t[q_mask] + np.sin(t[q_mask] / 6.0)
    panel = pd.DataFrame(
        {
            "m1": np.sin(t / 5.0) + t / 100.0,
            "m2": np.cos(t / 7.0),
            "q_target": q_target,
        },
        index=idx,
    )
    return mf.data.set_frequencies(
        panel,
        {"m1": "monthly", "m2": "monthly", "q_target": "quarterly"},
        frequency="mixed",
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
    assert {"preprocessed", "combined", "combination"}.issubset(table.columns)
    assert table["preprocessed"].eq(False).all()
    assert table["combined"].eq(False).all()
    assert table["combination"].isna().all()
    _assert_forecast_result_metadata(result, "panel_to_features")
    assert result.metadata["features"]["pca_components"] == 1
    json.dumps(result.to_dict())


def test_forecasting_runner_supports_multiple_horizons_and_target_dates() -> None:
    panel = _panel()

    result = mf.forecasting.run(
        panel,
        "ols",
        window=_window(),
        target="y",
        horizons=[1, 3],
        save_models=False,
    )
    table = result.to_frame()
    base = table.loc[~table["combined"].fillna(False).astype(bool)]

    assert set(base["horizon"]) == {1, 3}
    assert set(base["forecast_policy"]) == {"direct"}
    assert set(base["target"]) == {"y"}
    assert result.metadata["run"]["multi_horizon"] is True
    assert result.metadata["run"]["horizons"] == [1, 3]
    for _, row in base.iterrows():
        origin_pos = int(row["origin_pos"])
        horizon = int(row["horizon"])
        assert row["date"] == panel.index[origin_pos + horizon]
        assert np.isclose(row["actual"], panel["y"].iloc[origin_pos + horizon])


def test_forecasting_runner_supports_direct_average_policy() -> None:
    panel = _panel()

    result = mf.forecasting.run(
        panel,
        "ols",
        window=_window(),
        target="y",
        horizon=2,
        forecast_policy="direct_average",
        target_transform="growth",
        save_models=False,
    )
    table = result.to_frame()
    row = table.iloc[0]
    origin_pos = int(row["origin_pos"])
    y0 = panel["y"].iloc[origin_pos]
    y1 = panel["y"].iloc[origin_pos + 1]
    y2 = panel["y"].iloc[origin_pos + 2]

    assert set(table["forecast_policy"]) == {"direct_average"}
    assert set(table["horizon"]) == {2}
    assert row["date"] == panel.index[origin_pos + 2]
    assert np.isclose(row["actual"], ((y1 / y0 - 1.0) + (y2 / y1 - 1.0)) / 2.0)
    assert result.metadata["features"]["target_transform"] == "average_growth"


def test_forecasting_runner_supports_path_average_policy() -> None:
    panel = _panel()

    result = mf.forecasting.run(
        panel,
        "ols",
        window=_window(),
        target="y",
        horizon=2,
        forecast_policy="path_average",
        target_transform="change",
        save_models=False,
    )
    table = result.to_frame()
    row = table.iloc[0]
    origin_pos = int(row["origin_pos"])
    y0 = panel["y"].iloc[origin_pos]
    y1 = panel["y"].iloc[origin_pos + 1]
    y2 = panel["y"].iloc[origin_pos + 2]

    assert set(table["forecast_policy"]) == {"path_average"}
    assert set(table["horizon"]) == {2}
    assert row["date"] == panel.index[origin_pos + 2]
    assert np.isclose(row["actual"], ((y1 - y0) + (y2 - y1)) / 2.0)
    assert result.metadata["features"]["target_mode"] == "path"


def test_forecasting_runner_supports_recursive_policy_with_target_lags() -> None:
    panel = _panel(60)

    result = mf.forecasting.run(
        panel,
        "ols",
        window=_window(),
        target="y",
        horizon=3,
        forecast_policy="recursive",
        save_models=False,
    )
    table = result.to_frame()
    row = table.iloc[0]
    origin_pos = int(row["origin_pos"])

    assert set(table["forecast_policy"]) == {"recursive"}
    assert set(table["horizon"]) == {3}
    assert row["date"] == panel.index[origin_pos + 3]
    assert np.isclose(row["actual"], panel["y"].iloc[origin_pos + 3])
    assert result.metadata["run"]["future_feature_policy"] == "target_lags"
    assert result.metadata["forecast_policy"]["uses_observed_future_predictors"] is False
    assert "step_predictions" in row["params"]["recursive"]


@pytest.mark.parametrize(
    ("target_transform", "expected"),
    [
        ("level", lambda y0, yh: yh),
        ("change", lambda y0, yh: yh - y0),
        ("growth", lambda y0, yh: yh / y0 - 1.0),
        ("log_growth", lambda y0, yh: np.log(yh) - np.log(y0)),
    ],
)
def test_forecasting_runner_recursive_actual_matches_target_transform(
    target_transform: str,
    expected,
) -> None:
    panel = _panel(60)

    result = mf.forecasting.run(
        panel,
        "ols",
        window=_window(),
        target="y",
        horizon=3,
        forecast_policy="recursive",
        target_transform=target_transform,
        save_models=False,
    )
    row = result.to_frame().iloc[0]
    origin_pos = int(row["origin_pos"])
    y0 = panel["y"].iloc[origin_pos]
    yh = panel["y"].iloc[origin_pos + 3]

    assert np.isclose(row["actual"], expected(y0, yh))
    assert result.metadata["features"]["target_transform"] == target_transform


def test_forecasting_runner_supports_recursive_multiple_horizons() -> None:
    panel = _panel(72)

    result = mf.forecasting.run(
        panel,
        "ols",
        window=_window(),
        target="y",
        horizons=[1, 3],
        forecast_policy="iterated",
        save_models=False,
    )
    table = result.to_frame()
    base = table.loc[~table["combined"].fillna(False).astype(bool)]

    assert set(base["forecast_policy"]) == {"recursive"}
    assert set(base["horizon"]) == {1, 3}
    assert result.metadata["run"]["multi_horizon"] is True
    assert result.metadata["forecast_policy"]["method"] == "recursive"
    assert result.metadata["forecast_policy"]["horizons"] == [1, 3]
    for _, row in base.iterrows():
        origin_pos = int(row["origin_pos"])
        horizon = int(row["horizon"])
        assert row["date"] == panel.index[origin_pos + horizon]


def test_forecasting_runner_recursive_works_with_preprocessing_and_feature_policy() -> None:
    panel = _panel(72)
    pre = mf.preprocessing.preprocess_spec(
        transform="none",
        outliers="none",
        impute="mean",
        standardize="zscore",
        frame="keep",
    )
    features = mf.feature_engineering.feature_spec(
        target="y",
        predictors=[],
        lags=None,
        target_lags=(0, 1),
        add_time=True,
    )

    result = mf.forecasting.run(
        panel,
        "ridge",
        window=_window(),
        preprocessing=pre,
        preprocessing_policy=mf.window.stage_policy("origin_available"),
        features=features,
        feature_policy=mf.window.stage_policy("fit_window"),
        params={"ridge": {"alpha": 0.1}},
        model_selection={"ridge": None},
        horizon=2,
        forecast_policy="recursive",
        save_models=False,
    )
    table = result.to_frame()

    assert not table.empty
    assert table["preprocessed"].all()
    assert set(table["forecast_policy"]) == {"recursive"}
    assert result.metadata["stage_policies"]["preprocessing"]["scope"] == "origin_available"
    assert result.metadata["stage_policies"]["feature_engineering"]["scope"] == "fit_window"
    assert result.metadata["features"]["target_lags"] == [0, 1]


def test_forecasting_runner_recursive_saves_trained_model(tmp_path) -> None:
    panel = _panel(60)

    result = mf.forecasting.run(
        panel,
        "ridge",
        window=_window(),
        target="y",
        horizon=2,
        forecast_policy="recursive",
        model_selection=mf.model_selection.grid({"alpha": [0.01, 0.1]}),
        model_store=tmp_path / "trained_model",
    )
    row = result.to_frame()["stored_model"].dropna().iloc[0]
    metadata_path = Path(row["metadata_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata_path.exists()
    assert metadata["window"]["forecast_policy"] == "recursive"
    assert metadata["window"]["future_feature_policy"] == "target_lags"
    assert metadata["params"]["alpha"] in {0.01, 0.1}
    assert result.metadata["run"]["save_models"] is True


def test_forecasting_runner_combines_recursive_rows() -> None:
    panel = _panel(60)

    result = mf.forecasting.run(
        panel,
        ["ols", "ridge"],
        window=_window(),
        target="y",
        horizon=2,
        forecast_policy="recursive",
        params={"ridge": {"alpha": 0.1}},
        model_selection={"ols": None, "ridge": None},
        combination="mean",
        save_models=False,
    )
    table = result.to_frame()
    base = table.loc[table["model"].isin(["ols", "ridge"])]
    combined = table.loc[table["model"] == "combined_mean"]

    assert not combined.empty
    assert set(combined["forecast_policy"]) == {"recursive"}
    assert combined["combined"].all()
    assert {item["method"] for item in combined["combination"]} == {"mean"}
    for key, group in base.groupby(["date", "origin_pos", "horizon"], sort=False):
        prediction = combined.set_index(["date", "origin_pos", "horizon"]).loc[
            key,
            "prediction",
        ]
        assert np.isclose(prediction, group["prediction"].mean())


def test_forecasting_runner_rejects_recursive_target_lags_with_exog_features() -> None:
    panel = _panel()
    features = mf.feature_engineering.feature_spec(
        target="y",
        predictors=["x1"],
        target_lags=(0, 1),
        lags=(0, 1),
    )

    with pytest.raises(ValueError, match="predictors to be empty"):
        mf.forecasting.run(
            panel,
            "ols",
            window=_window(),
            features=features,
            horizon=2,
            forecast_policy="recursive",
            save_models=False,
        )


def test_forecasting_runner_rejects_recursive_target_lags_without_current_lag() -> None:
    panel = _panel()
    features = mf.feature_engineering.feature_spec(
        target="y",
        predictors=[],
        lags=None,
        target_lags=(1, 2),
    )

    with pytest.raises(ValueError, match="include 0"):
        mf.forecasting.run(
            panel,
            "ols",
            window=_window(),
            features=features,
            horizon=2,
            forecast_policy="recursive",
            save_models=False,
        )


def test_forecasting_runner_allows_recursive_observed_future_policy() -> None:
    panel = _panel()
    features = mf.feature_engineering.feature_spec(
        target="y",
        predictors=["x1"],
        target_lags=(0, 1),
        lags=(0, 1),
    )

    result = mf.forecasting.run(
        panel,
        "ols",
        window=_window(),
        features=features,
        horizon=2,
        forecast_policy="iterated",
        future_feature_policy="observed_future",
        save_models=False,
    )
    table = result.to_frame()

    assert not table.empty
    assert set(table["forecast_policy"]) == {"recursive"}
    assert result.metadata["run"]["future_feature_policy"] == "observed_future"
    assert result.metadata["forecast_policy"]["uses_observed_future_predictors"] is True


def test_forecasting_runner_supports_fit_aware_feature_steps() -> None:
    panel = _panel()
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        steps=[
            mf.feature_engineering.scale_step(name="scaled", include=False),
            mf.feature_engineering.pca_step(
                name="pc",
                input="scaled",
                n_components=1,
                min_train_size=12,
                include=False,
            ),
            mf.feature_engineering.lag_step(name="pc_lag", input="pc", lags=(0, 1)),
        ],
    )

    result = mf.forecasting.run(
        panel,
        "ridge",
        window=_window(),
        features=features,
        feature_policy=mf.window.stage_policy("fit_window"),
        params={"ridge": {"alpha": 0.1}},
        model_selection={"ridge": None},
        save_models=False,
    )
    table = result.to_frame()
    stage = next(item for item in result.metadata["stages"] if item["stage"] == "feature_engineering")

    assert not table.empty
    assert table["prediction"].notna().all()
    assert result.metadata["features"]["feature_steps"][0]["method"] == "scale"
    assert stage["metadata"]["feature_steps"][1]["fit_state"]["n_components"] == 1
    json.dumps(result.to_dict())


def test_forecasting_runner_supports_target_aware_feature_steps() -> None:
    panel = _panel()
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        steps=[
            mf.feature_engineering.scale_step(name="scaled", include=False),
            mf.feature_engineering.partial_least_squares_step(
                name="pls",
                input="scaled",
                n_components=1,
                min_train_size=12,
            ),
        ],
    )

    result = mf.forecasting.run(
        panel,
        "ridge",
        window=_window(),
        features=features,
        feature_policy=mf.window.stage_policy("fit_window"),
        params={"ridge": {"alpha": 0.1}},
        model_selection={"ridge": None},
        save_models=False,
    )
    table = result.to_frame()
    stage = next(item for item in result.metadata["stages"] if item["stage"] == "feature_engineering")

    assert not table.empty
    assert table["prediction"].notna().all()
    assert result.metadata["features"]["feature_steps"][1]["method"] == "partial_least_squares"
    assert stage["metadata"]["feature_steps"][1]["fit_state"]["target"] == "y_level_h1"


def test_forecasting_runner_supports_fit_aware_marx_step() -> None:
    panel = _panel()
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        steps=[
            mf.feature_engineering.marx_step(
                max_lag=2,
                scale_lags=True,
                min_train_size=12,
            )
        ],
    )

    result = mf.forecasting.run(
        panel,
        "ridge",
        window=_window(),
        features=features,
        feature_policy=mf.window.stage_policy("fit_window"),
        params={"ridge": {"alpha": 0.1}},
        model_selection={"ridge": None},
        save_models=False,
    )
    table = result.to_frame()
    stage = next(item for item in result.metadata["stages"] if item["stage"] == "feature_engineering")

    assert not table.empty
    assert table["prediction"].notna().all()
    assert result.metadata["features"]["feature_steps"][0]["method"] == "marx"
    assert stage["metadata"]["feature_steps"][0]["fit_state"]["scale_lags"] is True


def test_forecasting_runner_supports_fit_aware_hamilton_step() -> None:
    panel = _panel(72)
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        steps=[
            mf.feature_engineering.hamilton_step(
                columns=["x1"],
                h=2,
                p=2,
                min_train_size=12,
            )
        ],
    )

    result = mf.forecasting.run(
        panel,
        "ridge",
        window=_window(),
        features=features,
        feature_policy=mf.window.stage_policy("fit_window"),
        params={"ridge": {"alpha": 0.1}},
        model_selection={"ridge": None},
        save_models=False,
    )
    table = result.to_frame()
    stage = next(item for item in result.metadata["stages"] if item["stage"] == "feature_engineering")

    assert not table.empty
    assert table["prediction"].notna().all()
    assert result.metadata["features"]["feature_steps"][0]["method"] == "hamilton_filter"
    assert stage["metadata"]["feature_steps"][0]["fit_state"]["fit_policy"] == "fixed_fit_panel"
    assert stage["metadata"]["feature_steps"][0]["fit_state"]["fit_rows_by_column"]["x1"] >= 12


def test_forecasting_runner_supports_fit_aware_projection_steps() -> None:
    panel = _panel(72)
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        steps=[
            mf.feature_engineering.random_projection_step(
                name="rp",
                n_components=2,
                random_state=0,
                min_train_size=12,
                include=False,
            ),
            mf.feature_engineering.nystroem_step(
                name="nys",
                input="rp",
                n_components=2,
                random_state=0,
                min_train_size=12,
            ),
        ],
    )

    result = mf.forecasting.run(
        panel,
        "ridge",
        window=_window(),
        features=features,
        feature_policy=mf.window.stage_policy("fit_window"),
        params={"ridge": {"alpha": 0.1}},
        model_selection={"ridge": None},
        save_models=False,
    )
    table = result.to_frame()
    stage = next(item for item in result.metadata["stages"] if item["stage"] == "feature_engineering")

    assert not table.empty
    assert table["prediction"].notna().all()
    assert result.metadata["features"]["feature_steps"][0]["method"] == "random_projection"
    assert stage["metadata"]["feature_steps"][0]["fit_state"]["fit_policy"] == "fixed_fit_panel"
    assert stage["metadata"]["feature_steps"][1]["fit_state"]["kernel"] == "rbf"


def test_forecasting_runner_supports_deterministic_feature_steps() -> None:
    panel = _panel()
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        steps=[
            mf.feature_engineering.transform_step(
                name="dx1",
                transform="diff",
                columns=["x1"],
                include=False,
            ),
            mf.feature_engineering.lag_step(name="dx1_lag", input="dx1", lags=(1,)),
            mf.feature_engineering.interaction_step(name="cross", columns=["x1", "x2"]),
        ],
    )

    result = mf.forecasting.run(
        panel,
        "ridge",
        window=_window(),
        features=features,
        feature_policy=mf.window.stage_policy("fit_window"),
        params={"ridge": {"alpha": 0.1}},
        model_selection={"ridge": None},
        save_models=False,
    )
    table = result.to_frame()
    stage = next(item for item in result.metadata["stages"] if item["stage"] == "feature_engineering")

    assert not table.empty
    assert table["prediction"].notna().all()
    assert result.metadata["features"]["feature_steps"][0]["method"] == "transform"
    assert stage["metadata"]["feature_steps"][0]["fit_state"] is None
    assert [item["method"] for item in stage["metadata"]["feature_steps"]] == ["transform", "lag", "interaction"]


def test_forecasting_runner_smoke_preprocess_feature_chain_and_selection() -> None:
    panel = _panel(60)
    preprocessing = mf.preprocessing.preprocess_spec(
        transform="none",
        outliers="none",
        impute="mean",
        frame="keep",
    )
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        steps=[
            mf.feature_engineering.transform_step(
                name="dx2",
                transform="diff",
                columns=["x2"],
                include=False,
            ),
            mf.feature_engineering.scale_step(name="scaled", input="dx2", include=False),
            mf.feature_engineering.pca_step(
                name="pc",
                input="scaled",
                n_components=1,
                min_train_size=12,
                include=False,
            ),
            mf.feature_engineering.lag_step(name="pc_lag", input="pc", lags=(0, 1)),
            mf.feature_engineering.time_step(name="time", trend=True),
        ],
    )

    result = mf.forecasting.run(
        panel,
        "ridge",
        window=_window(),
        preprocessing=preprocessing,
        preprocessing_policy=mf.window.stage_policy("fit_window"),
        features=features,
        feature_policy=mf.window.stage_policy("fit_window"),
        model_selection=mf.model_selection.grid({"alpha": [0.01, 0.1]}),
        save_models=False,
    )
    table = result.to_frame()
    feature_stage = next(item for item in result.metadata["stages"] if item["stage"] == "feature_engineering")

    assert not table.empty
    assert table["prediction"].notna().all()
    assert result.metadata["features"]["feature_steps"][0]["method"] == "transform"
    assert result.metadata["model_selection"]["method"] == "grid"
    assert result.metadata["model_selection"]["param_grid"] == {"alpha": [0.01, 0.1]}
    assert feature_stage["metadata"]["feature_steps"][2]["fit_state"]["n_components"] == 1
    assert {selection["retuned"] for selection in table["model_selection"]} == {True}


def test_forecasting_runner_preserves_data_bundle_metadata() -> None:
    panel = _panel()
    bundle = mf.data.set_frequencies(
        panel,
        {"y": "monthly", "x1": "monthly", "x2": "monthly"},
        frequency="monthly",
    )
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0, 1),
    )

    result = mf.forecasting.run(
        bundle,
        "ols",
        window=_window(),
        features=features,
        save_models=False,
    )

    assert result.metadata["data"]["metadata_frequency"] == "monthly"
    assert result.metadata["data"]["native_frequency_counts"] == {"monthly": 3}


def test_forecasting_runner_supports_panel_input_dfm_mixed_frequency() -> None:
    bundle = _mixed_panel()
    window = mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=30),
        val=mf.window.val_last_block(size=6),
        test=mf.window.test_origins(
            first_origin="2002-09-30",
            horizon=1,
            step=3,
        ),
    )

    result = mf.forecasting.run(
        bundle,
        "dfm_mixed_mariano_murasawa",
        window=window,
        target="q_target",
        params={
            "dfm_mixed_mariano_murasawa": {
                "maxiter": 5,
                "tolerance": 1e-3,
            }
        },
        model_selection={"dfm_mixed_mariano_murasawa": None},
        save_models=False,
    )
    table = result.to_frame()

    assert not table.empty
    assert set(table["model_spec"]) == {"dfm_mixed_mariano_murasawa"}
    assert table["prediction"].notna().all()
    assert set(table["params"].map(lambda value: value["target"])) == {"q_target"}
    assert table["preprocessed"].eq(False).all()
    assert table["combined"].eq(False).all()
    assert table["combination"].isna().all()
    assert result.metadata["run"]["panel_model_runner"] is True
    _assert_forecast_result_metadata(result, "panel_model")
    assert result.metadata["data"]["frequency"] == "mixed"
    assert result.metadata["data"]["native_frequency_counts"] == {
        "monthly": 2,
        "quarterly": 1,
    }


def test_forecasting_runner_rejects_panel_model_with_feature_spec() -> None:
    bundle = _mixed_panel()
    features = mf.feature_engineering.feature_spec(
        target="q_target",
        horizon=1,
        predictors=["m1", "m2"],
        lags=(0, 1),
    )

    with pytest.raises(ValueError, match="panel-input models consume the panel"):
        mf.forecasting.run(
            bundle,
            "dfm_mixed_mariano_murasawa",
            window=_window(),
            target="q_target",
            features=features,
            save_models=False,
        )


def test_forecasting_runner_rejects_conflicting_panel_model_target() -> None:
    bundle = _mixed_panel()

    with pytest.raises(ValueError, match="runner target conflicts"):
        mf.forecasting.run(
            bundle,
            "dfm_mixed_mariano_murasawa",
            window=_window(),
            target="q_target",
            params={
                "dfm_mixed_mariano_murasawa": {
                    "target": "other_target",
                    "maxiter": 5,
                    "tolerance": 1e-3,
                }
            },
            model_selection={"dfm_mixed_mariano_murasawa": None},
            save_models=False,
        )


def test_forecasting_runner_supports_composite_dfm_midas_panel_model() -> None:
    bundle = _mixed_panel()
    window = mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=36),
        val=mf.window.val_last_block(size=6),
        test=mf.window.test_origins(first_origin="2003-06-30", horizon=1, step=3),
    )

    result = mf.forecasting.run(
        bundle,
        "dfm_unrestricted_midas",
        window=window,
        target="q_target",
        params={
            "dfm_unrestricted_midas": {
                "target": "q_target",
                "lag_columns": ["m1"],
                "lags": (0, 1, 2),
                "factor_lags": (0,),
                "maxiter": 10,
                "tolerance": 1e-3,
            }
        },
        model_selection={"dfm_unrestricted_midas": None},
        save_models=False,
    )
    table = result.to_frame()

    assert not table.empty
    assert table["model_spec"].eq("dfm_unrestricted_midas").all()
    assert table["prediction"].notna().all()
    assert table["actual"].notna().all()
    assert table["model_selection"].map(lambda value: value["retuned"]).eq(False).all()


def test_forecasting_runner_accepts_explicit_mixed_frequency_midas_feature_set() -> None:
    idx = pd.date_range("2000-01-01", periods=96, freq="MS", name="date")
    t = np.arange(len(idx), dtype=float)
    q_mask = idx.month.isin([3, 6, 9, 12])
    q_target = pd.Series(np.nan, index=idx, name="q_target")
    q_target.loc[q_mask] = 10.0 + 0.2 * t[q_mask] + np.sin(t[q_mask] / 6.0)
    panel = pd.DataFrame(
        {
            "m1": np.sin(t / 4.0) + t / 100.0,
            "m2": np.cos(t / 5.0),
            "q_target": q_target,
        },
        index=idx,
    )
    bundle = mf.data.set_frequencies(
        panel,
        {"m1": "monthly", "m2": "monthly", "q_target": "quarterly"},
        frequency="mixed",
    )
    X_midas = mf.feature_engineering.mixed_frequency_lags(
        bundle,
        target="q_target",
        columns=["m1", "m2"],
        lags=(0, 1, 2),
        target_frequency="quarterly",
        anchor_position="period_end",
        drop_missing=True,
    )
    y = panel["q_target"].reindex(X_midas.index).rename("q_target").to_frame()
    feature_set = mf.feature_engineering.FeatureSet(
        X=X_midas,
        y=y,
        metadata={"feature_engineering": {"method": "mixed_frequency_lags"}},
        feature_metadata=X_midas.attrs["macroforecast_feature_metadata"],
        target="q_target",
        targets=("q_target",),
        horizons=(1,),
        predictors=tuple(X_midas.columns),
    )

    result = mf.forecasting.run(
        feature_set,
        "midas_beta",
        window=mf.window.spec(
            estimation=mf.window.estimation_expanding(min_size=16),
            val=mf.window.val_last_block(size=4),
            test=mf.window.test_origins(horizon=1, step=2),
        ),
        params={"midas_beta": {"beta_params": (1.0, 2.0), "alpha": 0.1}},
        model_selection={"midas_beta": None},
        save_models=False,
    )
    table = result.to_frame()

    assert not table.empty
    assert table["model_spec"].eq("midas_beta").all()
    assert table["prediction"].notna().all()
    assert result.metadata["run"]["input_path"] == "feature_set"
    assert result.metadata["features"]["method"] == "mixed_frequency_lags"


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
    ridge_selection = table.loc[table["model"] == "ridge", "model_selection"].dropna().iloc[0]
    assert ridge_selection["window"] == "explicit_splits"
    assert ridge_selection["metadata"]["split_source"] == "explicit"


def test_forecasting_runner_adds_combination_rows() -> None:
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
        params={"ridge": {"alpha": 0.1}},
        combination="mean",
        save_models=False,
    )
    table = result.to_frame()
    base = table.loc[table["model"].isin(["ols", "ridge"])]
    combined = table.loc[table["model"] == "combined_mean"].copy()
    combined_indexed = combined.set_index(["date", "origin_pos", "horizon"])

    assert not combined.empty
    assert result.metadata["run"]["n_combinations"] == 1
    assert result.metadata["run"]["n_combination_forecasts"] == len(combined)
    assert result.metadata["combination"][0]["method"] == "mean"
    assert base["combined"].eq(False).all()
    assert base["combination"].isna().all()
    assert combined["combined"].all()
    assert combined["preprocessed"].eq(False).all()
    assert combined["model_selection"].isna().all()
    assert combined["stored_model"].isna().all()
    assert {item["method"] for item in combined["combination"]} == {"mean"}
    for key, group in base.groupby(["date", "origin_pos", "horizon"], sort=False):
        assert np.isclose(
            combined_indexed.loc[key, "prediction"],
            group["prediction"].mean(),
        )


def test_forecasting_runner_supports_named_combination_specs() -> None:
    panel = _panel()
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0, 1),
    )

    result = mf.forecasting.run(
        panel,
        {"linear": "ols", "penalized": "ridge"},
        window=_window(),
        features=features,
        params={"penalized": {"alpha": 0.1}},
        combination={
            "combo": {
                "method": "dmspe",
                "models": ["linear", "penalized"],
                "discount": 0.95,
            }
        },
        save_models=False,
    )
    table = result.to_frame()

    assert "combo" in set(table["model"])
    assert result.metadata["combination"][0]["name"] == "combo"
    assert result.metadata["combination"][0]["params"]["discount"] == 0.95


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
        model_selection={"ridge": None},
    )
    table = result.to_frame()

    assert table["model_selection"].isna().all()
    assert result.metadata["models"][0]["spec"]["params"]["alpha"] == 0.1


def test_forecasting_runner_records_fixed_and_selected_model_params() -> None:
    panel = _panel()
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0, 1),
    )

    fixed = mf.forecasting.run(
        panel,
        "ridge",
        window=_window(),
        features=features,
        params={"ridge": {"alpha": 0.1}},
        model_selection={"ridge": None},
        save_models=False,
    )
    fixed_table = fixed.to_frame()

    assert fixed.metadata["models"][0]["spec"]["params"] == {"alpha": 0.1}
    assert {row["alpha"] for row in fixed_table["params"]} == {0.1}

    selected = mf.forecasting.run(
        panel,
        "ridge",
        window=_window(),
        features=features,
        params={"ridge": {"fit_intercept": False}},
        model_selection={"ridge": mf.model_selection.grid({"alpha": [0.1]})},
        save_models=False,
    )
    selected_params = selected.to_frame()["params"].iloc[0]

    assert selected_params == {"fit_intercept": False, "alpha": 0.1}


def test_forecasting_runner_accepts_direct_dict_valued_model_params() -> None:
    panel = _panel()
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0, 1),
    )
    bagging_spec = mf.model_ensemble.get_model_ensemble(
        "bagging",
        params={"n_estimators": 2},
    )

    result = mf.forecasting.run(
        panel,
        bagging_spec,
        window=_window(),
        features=features,
        params={"base_params": {"alpha": 0.2}},
        model_selection={"bagging": None},
        save_models=False,
    )
    recorded = result.metadata["models"][0]["spec"]["params"]

    assert recorded == {"n_estimators": 2, "base_params": {"alpha": 0.2}}
    assert result.to_frame()["params"].iloc[0] == recorded


def test_forecasting_runner_rejects_unknown_model_keyed_options() -> None:
    panel = _panel()
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0, 1),
    )

    with pytest.raises(ValueError, match="model_selection contains keys"):
        mf.forecasting.run(
            panel,
            "ridge",
            window=_window(),
            features=features,
            model_selection={"typo": None},
            save_models=False,
        )

    with pytest.raises(ValueError, match="params looks model-keyed"):
        mf.forecasting.run(
            panel,
            "ridge",
            window=_window(),
            features=features,
            params={"typo": {"alpha": 0.1}},
            model_selection={"ridge": None},
            save_models=False,
        )

    with pytest.raises(ValueError, match="preset contains keys"):
        mf.forecasting.run(
            panel,
            "ridge",
            window=_window(),
            features=features,
            preset={"typo": "small"},
            model_selection={"ridge": None},
            save_models=False,
        )


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
        model_selection={"pls": None},
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
        model_selection=mf.model_selection.grid({"alpha": [0.01, 0.1]}),
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
        model_selection={"scaled_pca": None},
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
        model_selection={"supervised_pca": None},
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
        model_selection={"supervised_scaled_pca": None},
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
        model_selection={"svr": None},
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
        model_selection={"random_forest": None},
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
        model_selection={"quantile_regression_forest": None},
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
        model_selection={"ar": None, "bagging": None},
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
        model_selection={model: None},
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
        model_selection={"nn": None},
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
        model_selection={"ridge": None},
    )
    table = result.to_frame()

    assert set(table["model"]) == {"ridge"}
    assert table["prediction"].notna().all()
    assert {"preprocessed", "combined", "combination"}.issubset(table.columns)
    assert table["preprocessed"].eq(False).all()
    assert table["combined"].eq(False).all()
    assert table["combination"].isna().all()
    _assert_forecast_result_metadata(result, "feature_set")
    assert result.metadata["features"]["spec"]["target"] == "y"
    assert result.metadata["features"]["output"]["n_features"] == feature_set.X.shape[1]


def test_forecast_result_json_export_has_stable_schema(tmp_path: Path) -> None:
    panel = _panel()
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
        features=features,
        save_models=False,
    )
    output_path = tmp_path / "forecast_result.json"
    text = result.to_json(output_path)
    payload = json.loads(text)
    from_disk = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload == from_disk
    assert payload["metadata"]["metadata_schema"]["kind"] == "forecast_result"
    assert payload["metadata"]["metadata_schema"]["version"] == 1
    assert payload["metadata"]["run"]["input_path"] == "panel_to_features"
    assert isinstance(payload["forecasts"][0]["date"], str)


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
        model_selection={"macro_random_forest": None},
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
        model_selection=mf.model_selection.grid({"alpha": [0.01, 0.1]}),
    )
    table = result.to_frame()

    assert not table.empty
    assert result.metadata["window"]["test"]["step"] == "2ME"
    assert result.metadata["window"]["estimation"]["retrain_every"] == 2
    assert {row["test_step"] for row in table["window"]} == {"2ME"}
    assert {bool(row["retrain"]) for row in table["window"]} == {False, True}
    assert {bool(row["retune"]) for row in table["window"]} == {False, True}
    assert {selection["retuned"] for selection in table["model_selection"]} == {False, True}


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


def test_forecasting_runner_records_preloop_stage_fit_as_initial_update() -> None:
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
        pca_components=1,
    )

    result = mf.forecasting.run(
        panel,
        "ols",
        window=_window(),
        preprocessing=pre,
        preprocessing_policy=mf.window.stage_policy("full_panel"),
        features=features,
        feature_policy=mf.window.stage_policy("full_panel"),
        save_models=False,
    )

    assert _stage_updates(result, "preprocessing")[:2] == [True, False]
    assert _stage_updates(result, "feature_engineering")[:2] == [True, False]


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


def test_forecasting_runner_accepts_variance_prediction_from_x_test() -> None:
    panel = _panel()
    captured = {}

    class XVarianceFit:
        def predict(self, X):
            return np.full(len(X), 2.0)

        def predict_variance(self, X):
            if not isinstance(X, pd.DataFrame):
                raise TypeError("X variance requires a DataFrame")
            captured["columns"] = tuple(X.columns)
            return np.full(len(X), 0.5)

    def x_variance_model(X, y):
        return XVarianceFit()

    spec = mf.models.ModelSpec(
        name="x_variance",
        family="test",
        fit_func=x_variance_model,
    )
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0,),
    )

    result = mf.forecasting.run(panel, spec, window=_window(), features=features)
    table = result.to_frame()

    assert set(table["prediction"]) == {2.0}
    assert set(table["variance_prediction"]) == {0.5}
    assert captured["columns"] == ("x1_lag0", "x2_lag0")


def test_forecasting_runner_rejects_misaligned_prediction_index() -> None:
    panel = _panel()

    class BadIndexFit:
        def predict(self, X):
            bad_index = pd.date_range("1990-01-31", periods=len(X), freq="ME")
            return pd.Series(np.ones(len(X)), index=bad_index)

    def bad_index_model(X, y):
        return BadIndexFit()

    spec = mf.models.ModelSpec(
        name="bad_index",
        family="test",
        fit_func=bad_index_model,
    )
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0,),
    )

    with pytest.raises(ValueError, match="prediction index does not match X_test"):
        mf.forecasting.run(
            panel,
            spec,
            window=_window(),
            features=features,
            save_models=False,
        )


def test_forecasting_runner_accepts_positional_prediction_series() -> None:
    panel = _panel()

    class PositionalFit:
        def predict(self, X):
            return pd.Series(np.full(len(X), 2.0))

    def positional_model(X, y):
        return PositionalFit()

    spec = mf.models.ModelSpec(
        name="positional",
        family="test",
        fit_func=positional_model,
    )
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0,),
    )

    result = mf.forecasting.run(
        panel,
        spec,
        window=_window(),
        features=features,
        save_models=False,
    )
    table = result.to_frame()

    assert not table.empty
    assert table["prediction"].eq(2.0).all()
    assert table["prediction"].notna().all()


def test_forecasting_runner_rejects_misaligned_quantile_prediction_index() -> None:
    panel = _panel()

    class BadQuantileFit:
        def predict(self, X):
            return np.ones(len(X))

        def predict_quantiles(self, X):
            bad_index = pd.date_range("1990-01-31", periods=len(X), freq="ME")
            return pd.DataFrame({"0.5": np.ones(len(X))}, index=bad_index)

    def bad_quantile_model(X, y):
        return BadQuantileFit()

    spec = mf.models.ModelSpec(
        name="bad_quantile",
        family="test",
        fit_func=bad_quantile_model,
    )
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0,),
    )

    with pytest.raises(
        ValueError,
        match="quantile prediction index does not match X_test",
    ):
        mf.forecasting.run(
            panel,
            spec,
            window=_window(),
            features=features,
            save_models=False,
        )


def test_forecasting_runner_rejects_missing_panel_target() -> None:
    panel = _panel()

    class FakePanelFit:
        def predict(self, X):
            return np.ones(len(X))

    def fake_panel_model(bundle, *, target=None):
        return FakePanelFit()

    spec = mf.models.ModelSpec(
        name="fake_panel",
        family="test",
        fit_func=fake_panel_model,
        input_kind="panel",
        default_params={"target": None},
    )

    with pytest.raises(ValueError, match="target 'missing' is not present"):
        mf.forecasting.run(
            panel,
            spec,
            window=_window(),
            target="missing",
            save_models=False,
        )


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
    spec = mf.forecasting.combination_spec(
        "discounted_mspe",
        name="combo",
        models=["m1", "m2"],
        discount=0.95,
    )
    assert spec.method == "dmspe"
    assert spec.name == "combo"
    assert spec.models == ("m1", "m2")


def _stage_updates(result: mf.forecasting.ForecastResult, stage: str) -> list[bool]:
    return [
        bool(record["updated"])
        for record in result.metadata["stages"]
        if record["stage"] == stage
    ]


def _assert_forecast_result_metadata(
    result: mf.forecasting.ForecastResult,
    input_path: str,
) -> None:
    metadata = result.metadata
    expected_keys = {
        "metadata_schema",
        "run",
        "data",
        "window",
        "stage_policies",
        "preprocessing",
        "features",
        "forecast_policy",
        "model_selection",
        "combination",
        "models",
        "stages",
    }
    assert expected_keys.issubset(metadata)
    assert metadata["metadata_schema"]["kind"] == "forecast_result"
    assert metadata["metadata_schema"]["version"] == 1
    assert metadata["metadata_schema"]["input_path"] == input_path
    assert metadata["run"]["input_path"] == input_path
    assert metadata["run"]["panel_model_runner"] is (input_path == "panel_model")
    assert metadata["metadata_schema"]["forecast_table_columns"] == [
        "date",
        "origin",
        "origin_pos",
        "horizon",
        "forecast_policy",
        "target_transform",
        "target",
        "model",
        "model_spec",
        "prediction",
        "variance_prediction",
        "quantile_predictions",
        "actual",
        "params",
        "model_selection",
        "stored_model",
        "window",
        "preprocessed",
        "combined",
        "combination",
    ]
    assert metadata["metadata_schema"]["stage_record_columns"] == [
        "stage",
        "origin",
        "origin_pos",
        "updated",
        "fit_start",
        "fit_end",
        "test_start",
        "test_end",
        "metadata",
    ]
