from __future__ import annotations

import numpy as np
import pandas as pd

import macroforecast as mf


def _panel(n: int = 36) -> pd.DataFrame:
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    x = np.linspace(1.0, 3.0, n)
    return pd.DataFrame(
        {
            "target": 2.0 + x,
            "x": x,
            "z": np.sin(np.arange(n) / 3.0),
        },
        index=idx,
    )


def _window() -> mf.window.WindowSpec:
    return mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=18),
        val=mf.window.val_last_block(size=6),
        test=mf.window.test_origins(horizon=1, step=6),
    )


def add_spread(panel: pd.DataFrame, *, metadata=None, scale: float = 1.0) -> pd.DataFrame:
    out = panel.copy()
    out["spread"] = (out["target"] - out["x"]) * scale
    return out


def square_feature(panel: pd.DataFrame, *, metadata=None, suffix: str = "sq") -> pd.DataFrame:
    column = panel.columns[0]
    return pd.DataFrame({f"{column}_{suffix}": panel[column] ** 2}, index=panel.index)


class MeanFit:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.value)


def mean_model(X: pd.DataFrame, y: pd.Series, *, offset: float = 0.0) -> MeanFit:
    return MeanFit(float(pd.Series(y).mean()) + offset)


def blend(forecasts: pd.DataFrame, *, actual: pd.Series, weight: float = 0.5) -> pd.Series:
    return weight * forecasts.iloc[:, 0] + (1.0 - weight) * forecasts.iloc[:, -1]


def ordered_offset_search(
    *,
    model,
    X,
    y,
    splits,
    metric,
    fixed_params,
    evaluate_candidate,
    values,
    **_,
):
    return (
        [
            evaluate_candidate(
                model,
                X,
                y,
                splits,
                metric,
                fixed_params,
                {"offset": value},
                trial,
            )
            for trial, value in enumerate(values)
        ],
        {"custom_runtime": {"evaluated": len(values)}},
    )


def last_fit_half(index: pd.Index, *, item: dict, policy: mf.window.StagePolicy) -> pd.Index:
    fit_idx = item["fit_idx"]
    keep = fit_idx[len(fit_idx) // 2 :]
    return index[keep]


def test_custom_dataset_and_preprocessing_helpers() -> None:
    frame = _panel().reset_index()
    bundle = mf.data.custom_dataset(
        frame,
        date="date",
        dataset="unit_custom",
        frequency="monthly",
        transform_codes={"target": 1, "x": 1, "z": 1},
    )

    assert isinstance(bundle, mf.data.DataBundle)
    assert bundle.metadata["dataset"] == "unit_custom"
    assert bundle.metadata["transform_codes"]["target"] == 1

    direct = mf.preprocessing.custom_preprocess(bundle, add_spread, name="spread", scale=2.0)
    assert "spread" in direct.panel.columns
    assert direct.metadata["custom_preprocess"]["name"] == "spread"

    spec = mf.preprocessing.preprocess_spec(
        transform="none",
        outliers="none",
        impute="none",
        standardize="none",
        frame="keep",
        custom_steps=[mf.preprocessing.custom_preprocess_step("spread", add_spread, scale=1.0)],
    )
    processed = spec.fit_transform(bundle)

    assert "spread" in processed.panel.columns
    assert processed.metadata["custom_preprocess_steps"][0]["name"] == "spread"
    assert processed.metadata["preprocess_spec"]["options"]["custom_steps"][0]["name"] == "spread"


def test_custom_feature_helpers_and_runner_safe_step() -> None:
    panel = _panel()
    direct = mf.feature_engineering.custom_features(
        panel,
        square_feature,
        columns=["x"],
        name="x_square",
        suffix="sq",
    )
    assert list(direct.columns) == ["x_sq"]
    assert direct.attrs["macroforecast_metadata"]["feature_engineering_custom"]["name"] == "x_square"

    spec = mf.feature_engineering.feature_spec(
        target="target",
        horizon=1,
        predictors=["x", "z"],
        steps=[
            mf.feature_engineering.custom_step(
                "x_square",
                square_feature,
                columns=["x"],
                suffix="sq",
            )
        ],
    )
    features = spec.fit_transform(panel)

    assert "x_sq" in features.X.columns
    assert features.feature_metadata.loc[features.feature_metadata["feature"] == "x_sq", "operation"].iloc[0] == "custom"
    assert spec.to_dict()["feature_steps"][0]["func"].endswith("square_feature")


def test_custom_model_combination_and_stage_policy_work_in_runner() -> None:
    panel = _panel()
    features = mf.feature_engineering.feature_spec(
        target="target",
        horizon=1,
        predictors=["x", "z"],
        lags=(0,),
    )
    model = {
        "ols": "ols",
        "mean_custom": mf.models.custom_model(
            "mean_custom",
            mean_model,
            default_params={"offset": 0.1},
        ),
    }
    tuned_custom = mf.models.custom_model(
        "mean_tuned_custom",
        mean_model,
        default_preset="small",
        search_spaces={"small": {"offset": (-0.1, 0.0, 0.1)}},
    )
    assert tuned_custom.search_space() == {"offset": (-0.1, 0.0, 0.1)}

    selection_policy = mf.window.custom_stage_policy(last_fit_half)
    result = mf.forecasting.run(
        panel,
        model,
        window=_window(),
        features=features,
        selection_policy=selection_policy,
        combination=mf.forecasting.custom_combination("blend", blend, weight=0.25),
    )
    table = result.to_frame()

    assert "blend" in set(table["model"])
    assert table.loc[table["model"] == "blend", "combined"].all()
    assert result.metadata["stage_policies"]["selection"]["scope"] == "custom"
    assert result.metadata["combination"][0]["callable"].endswith("blend")


def test_custom_extension_flow_runs_from_data_to_output(tmp_path) -> None:
    bundle = mf.data.custom_dataset(
        _panel(48).reset_index(),
        date="date",
        dataset="custom_flow",
        frequency="monthly",
        transform_codes={"target": 1, "x": 1, "z": 1},
    )
    preprocessing = mf.preprocessing.preprocess_spec(
        transform="none",
        outliers="none",
        impute="none",
        standardize="none",
        frame="keep",
        custom_steps=[mf.preprocessing.custom_preprocess_step("spread", add_spread, scale=1.0)],
    )
    features = mf.feature_engineering.feature_spec(
        target="target",
        horizon=1,
        predictors=["x", "z", "spread"],
        lags=(0,),
        steps=[
            mf.feature_engineering.custom_step(
                "x_square",
                square_feature,
                columns=["x"],
                suffix="sq",
            )
        ],
    )
    mean_tuned = mf.models.custom_model(
        "mean_tuned",
        mean_model,
        default_params={"offset": 0.0},
    )
    search = mf.selection.custom_search(
        "ordered_offset",
        ordered_offset_search,
        values=(-0.1, 0.0, 0.1),
    )
    result = mf.forecasting.run(
        bundle,
        {"ols": "ols", "mean_tuned": mean_tuned},
        window=_window(),
        preprocessing=preprocessing,
        features=features,
        selection={"ols": None, "mean_tuned": search},
        selection_policy=mf.window.custom_stage_policy(last_fit_half),
        combination=mf.forecasting.custom_combination("blend", blend, weight=0.5),
        model_store=tmp_path / "trained_model",
    )

    forecast_table = result.to_frame()
    assert {"ols", "mean_tuned", "blend"}.issubset(set(forecast_table["model"]))
    selected = forecast_table.loc[forecast_table["model"] == "mean_tuned", "selection"].dropna().iloc[0]
    assert selected["method"] == "custom"
    assert selected["metadata"]["custom_runtime"] == {"evaluated": 3}

    def mean_bias(y_true, y_pred):
        return float(pd.Series(y_pred).sub(pd.Series(y_true)).mean())

    scores = result.evaluate(metrics=("mse", mean_bias))
    assert "mean_bias" in scores.columns

    test = mf.tests.custom_test(
        "flow_bias_test",
        lambda values: {"statistic": float(pd.Series(values).mean()), "p_value": 0.5, "n_obs": len(values)},
        forecast_table["prediction"] - forecast_table["actual"],
    )
    assert test.metadata["custom"] is True

    processed = preprocessing.fit_transform(bundle)
    feature_set = features.fit_transform(processed.panel)
    design = pd.concat([feature_set.X, feature_set.y.iloc[:, 0].rename("target_h1")], axis=1).dropna()
    fit = mf.models.ridge(design[feature_set.X.columns], design["target_h1"])

    interpretation = mf.interpretation.custom_interpretation(
        fit,
        feature_set.X.dropna().iloc[:5],
        lambda model, X, **_: {"mean_prediction": float(model.predict(X).mean())},
        name="mean_prediction",
    )
    feature_diag = mf.feature_diagnostic.custom_feature_diagnostic(
        feature_set,
        lambda X, **_: {"n_features": X.shape[1], "missing_cells": int(X.isna().sum().sum())},
        name="shape_check",
    )
    forecast_diag = mf.forecast_diagnostic.custom_forecast_diagnostic(
        result,
        lambda forecasts, **_: forecasts.groupby("model", as_index=False)["prediction"].mean(),
        name="mean_prediction_by_model",
    )
    manifest = mf.output.write_artifacts(
        {
            "forecast_result": result,
            "scores": scores,
            "custom_test": test.to_dict(),
            "custom_interpretation": interpretation,
            "custom_feature_diagnostic": feature_diag,
            "custom_forecast_diagnostic": forecast_diag,
        },
        tmp_path / "artifacts",
    )

    assert "custom_interpretation.csv" in manifest.artifacts
    assert "custom_feature_diagnostic.csv" in manifest.artifacts
    assert "custom_forecast_diagnostic.csv" in manifest.artifacts
