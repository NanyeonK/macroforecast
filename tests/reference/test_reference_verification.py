from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf

pytestmark = pytest.mark.reference


def _loss_panel() -> pd.DataFrame:
    rng = np.random.default_rng(20260601)
    rows = []
    for origin in range(48):
        common = 0.02 * np.sin(origin / 4.0)
        for model, base in (("benchmark", 0.70), ("candidate", 0.42), ("weak", 0.55)):
            rows.append(
                {
                    "target": "y",
                    "horizon": 1,
                    "origin": origin,
                    "model_id": model,
                    "squared_error": base + common + rng.normal(0.0, 0.02),
                }
            )
    return pd.DataFrame(rows)


def _reference_panel() -> pd.DataFrame:
    return mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=8, freq="MS"),
                "target": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0],
                "x1": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
                "x2": [20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0],
            }
        ),
        date="date",
        metadata={"dataset": "reference", "source_family": "synthetic", "frequency": "monthly"},
    )


def _reference_processed() -> mf.preprocessing.PreprocessedData:
    bundle = mf.data.DataBundle(
        _reference_panel(),
        {"dataset": "reference", "source_family": "synthetic", "frequency": "monthly"},
    )
    data_spec = mf.data.spec(
        bundle,
        target="target",
        horizons=[1, 2],
        predictors=["x1", "x2"],
    )
    return mf.preprocessing.reprocess(
        data_spec,
        transform="none",
        outliers="none",
        impute="none",
        frame="keep",
    )


def _reference_forecast_panel(n: int = 36) -> pd.DataFrame:
    index = pd.date_range("2018-01-31", periods=n, freq="ME", name="date")
    t = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "y": 1.0 + 0.2 * t + np.sin(t / 4.0),
            "x1": t,
            "x2": np.cos(t / 5.0),
        },
        index=index,
    )


def _reference_window() -> mf.window.WindowSpec:
    return mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=18),
        val=mf.window.val_last_block(size=4),
        test=mf.window.test_origins(horizon=1, step=6),
    )


def _manual_marx(panel: pd.DataFrame, *, columns: list[str], max_lag: int) -> pd.DataFrame:
    lag_values = tuple(range(1, max_lag + 1))
    lag_matrix = pd.DataFrame(
        {
            f"{column}_lag{lag}": panel[column].shift(lag)
            for lag in lag_values
            for column in columns
        },
        index=panel.index,
    )
    result = pd.DataFrame(index=panel.index)
    for column in columns:
        for lag_value in lag_values:
            lag_columns = [f"{column}_lag{step}" for step in range(1, lag_value + 1)]
            result[f"{column}_ma{lag_value}_lag1"] = lag_matrix.loc[:, lag_columns].mean(
                axis=1,
                skipna=False,
            )
    result.index.name = "date"
    return result


def test_dm_test_is_antisymmetric_reference_anchor() -> None:
    loss_a = pd.Series([0.3, 0.5, 0.4, 0.6, 0.7, 0.4, 0.5, 0.3])
    loss_b = pd.Series([0.5, 0.7, 0.5, 0.8, 0.9, 0.6, 0.7, 0.5])

    ab = mf.tests.dm_test(loss_a, loss_b, correction="none")
    ba = mf.tests.dm_test(loss_b, loss_a, correction="none")

    assert ab.statistic == pytest.approx(-ba.statistic)
    assert ab.p_value == pytest.approx(ba.p_value)
    assert ab.n_obs == ba.n_obs == 8


def test_blocked_reality_check_reference_anchor() -> None:
    result = mf.tests.blocked_oob_reality_check(
        _loss_panel(),
        benchmark="benchmark",
        alpha=0.1,
        n_boot=40,
        block_length=4,
        random_state=123,
    )

    candidate = result.set_index("model").loc["candidate"]
    assert result.attrs["macroforecast_metadata_schema"]["kind"] == "blocked_oob_reality_check"
    assert candidate["mean_diff"] > 0.0
    assert 0.0 <= candidate["p_value"] <= 1.0
    assert bool(candidate["decision"]) is True


def test_iterative_mcs_reference_anchor() -> None:
    result = mf.tests.iterative_model_confidence_set(
        _loss_panel(),
        alpha=0.1,
        n_boot=40,
        block_length=4,
        random_state=123,
    )
    included = result["mcs_inclusion"][0]["models"]
    rejected = result["mcs_rejections"][0]["models"]

    assert result["metadata_schema"]["kind"] == "iterative_model_confidence_set"
    assert "candidate" in included
    assert "benchmark" in rejected
    assert result["iteration_path"][0]["eliminated_model"] == "benchmark"
    assert result["statistic"] == "max"
    assert result["r_reference"] == "MCS/R/MCSprocedure.R::MCSprocedure"
    json.dumps(result)


def test_reporting_output_metadata_reference_anchor(tmp_path) -> None:
    scores = pd.DataFrame(
        {
            "model": ["candidate", "benchmark"],
            "rmse": [0.42, 0.70],
            "r2_oos": [0.12, -0.05],
        }
    )
    table = mf.reporting.report_table(
        scores,
        columns=("model", "rmse", "r2_oos"),
        rename={"model": "Model", "rmse": "RMSE", "r2_oos": "R2 OOS"},
        percent_columns=("R2 OOS",),
        precision=2,
        caption="Reference accuracy table",
    )
    bundle = mf.output.bundle_outputs(
        metadata={"reference_suite": "core"},
        extra={"accuracy_table": table.data},
    )
    manifest = mf.output.write_artifacts(
        mf.output.select_outputs(bundle, objects=("accuracy_table", "metadata")),
        tmp_path,
        formats=("json",),
        include_provenance=False,
    )

    assert table.data.attrs["macroforecast_metadata_schema"]["kind"] == "report_table"
    assert "\\caption{Reference accuracy table}" in table.to_latex()
    assert (tmp_path / "accuracy_table.json").exists()
    assert manifest.records[0].metadata["path_exists"] is True


def test_direct_average_and_path_targets_reference_anchor() -> None:
    processed = _reference_processed()

    direct = mf.feature_engineering.direct_target(processed, horizon=2, transform="change")
    average = mf.feature_engineering.average_target(processed, horizon=2, transform="change")
    path = mf.feature_engineering.path_targets(processed, horizon=2, transform="change")

    assert direct["target_change_h2"].iloc[0] == pytest.approx(3.0)
    assert average["target_average_change_h2"].iloc[0] == pytest.approx((1.0 + 2.0) / 2.0)
    assert path.loc[path.index[0], ["target_change_step1", "target_change_step2"]].tolist() == [
        1.0,
        2.0,
    ]
    direct_meta = direct.attrs["macroforecast_target_metadata"].set_index("target_column")
    path_meta = path.attrs["macroforecast_target_metadata"].set_index("target_column")
    assert direct_meta.loc["target_change_h2", "formula"] == "target[t+2] - target[t]"
    assert path_meta.loc["target_change_step2", "aggregation"] == "average_step_forecasts_in_evaluation"


def test_marx_moving_average_loop_reference_anchor() -> None:
    processed = _reference_processed()

    matrix = mf.feature_engineering.feature_matrix(
        processed,
        specification="MARX",
        columns=["x1", "x2"],
        max_lag=3,
    )
    expected = _manual_marx(processed.panel, columns=["x1", "x2"], max_lag=3).add_prefix("MARX__")

    pd.testing.assert_frame_equal(matrix, expected)
    assert matrix["MARX__x1_ma3_lag1"].iloc[3] == pytest.approx(np.mean([12.0, 11.0, 10.0]))


def test_maf_variable_specific_pca_reference_anchor() -> None:
    processed = _reference_processed()

    maf = mf.feature_engineering.maf_features(
        processed,
        columns=["x1", "x2"],
        max_lag=2,
        n_components=2,
        fit_policy="full_sample",
        min_train_size=3,
        scale=False,
        warn_full_sample=False,
    )
    x1_lag_panel = mf.feature_engineering.lag(processed, columns=["x1"], lags=[0, 1, 2])
    expected_x1 = mf.feature_engineering.pca_features(
        x1_lag_panel,
        n_components=2,
        fit_policy="full_sample",
        min_train_size=3,
        scale=False,
        prefix="x1_maf",
        warn_full_sample=False,
    )

    pd.testing.assert_frame_equal(maf.loc[:, ["x1_maf1", "x1_maf2"]], expected_x1)
    stage = maf.attrs["macroforecast_metadata"]["feature_engineering_maf"]
    assert stage["lags"] == [0, 1, 2]
    assert "not global PCA" in stage["note"]


def test_midas_weight_shape_reference_anchor() -> None:
    idx = pd.date_range("2000-01-01", periods=10, freq="MS")
    lagged = pd.DataFrame(
        {
            "x_lag0": np.arange(10, dtype=float),
            "x_lag1": np.arange(10, dtype=float) + 1.0,
            "x_lag2": np.arange(10, dtype=float) + 2.0,
            "x_lag3": np.arange(10, dtype=float) + 3.0,
        },
        index=idx,
    )
    target = pd.Series(np.arange(10, dtype=float), index=idx, name="y")

    almon = mf.models.midas_almon(lagged, target, polynomial_order=2, theta=(0.1, -0.02))
    positions = np.arange(1, 5, dtype=float)
    raw = 0.1 * positions + -0.02 * positions**2
    expected_almon = np.exp(raw - raw.max())
    expected_almon = expected_almon / expected_almon.sum()

    beta = mf.models.midas_beta(lagged, target, beta_params=(1.5, 2.0))
    z = (np.arange(1, 5, dtype=float) - 1.0) / 3.0
    z[0] += np.finfo(float).eps
    z[-1] -= np.finfo(float).eps
    expected_beta = z**0.5 * (1.0 - z)
    expected_beta = expected_beta / expected_beta.sum()

    step = mf.models.midas_step(lagged, target, step_bounds=(2,), step_weights=(1.0, 3.0))
    expected_step = np.asarray([1.0, 1.0, 3.0, 3.0], dtype=float)
    expected_step = expected_step / expected_step.sum()

    np.testing.assert_allclose(almon.metadata["weights"]["x"], expected_almon)
    np.testing.assert_allclose(beta.metadata["weights"]["x"], expected_beta)
    np.testing.assert_allclose(step.metadata["weights"]["x"], expected_step)


def test_runner_stage_policy_leakage_reference_anchor() -> None:
    panel = _reference_forecast_panel()
    window = _reference_window()
    pre = mf.preprocessing.preprocess_spec(
        transform="none",
        outliers="none",
        impute="none",
        standardize="zscore",
        standardize_columns=("x1", "x2"),
        standardize_ddof=0,
        frame="keep",
    )
    features = mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0,),
        pca_components=1,
    )

    fit_window_result = mf.forecasting.run(
        panel,
        "ols",
        window=window,
        preprocessing=pre,
        preprocessing_policy=mf.window.stage_policy("fit_window"),
        features=features,
        feature_policy=mf.window.stage_policy("fit_window"),
        save_models=False,
    )
    full_panel_result = mf.forecasting.run(
        panel,
        "ols",
        window=window,
        preprocessing=pre,
        preprocessing_policy=mf.window.stage_policy("full_panel"),
        features=features,
        feature_policy=mf.window.stage_policy("full_panel"),
        save_models=False,
    )
    first_item = next(window.iter_origins(panel.index))
    first_fit_end = pd.Timestamp(first_item["row"]["fit_end"]).strftime("%Y-%m-%d")
    fit_window_pre = [
        record
        for record in fit_window_result.metadata["stages"]
        if record["stage"] == "preprocessing"
    ]
    fit_window_features = [
        record
        for record in fit_window_result.metadata["stages"]
        if record["stage"] == "feature_engineering"
    ]
    full_pre = [
        record
        for record in full_panel_result.metadata["stages"]
        if record["stage"] == "preprocessing"
    ]
    full_features = [
        record
        for record in full_panel_result.metadata["stages"]
        if record["stage"] == "feature_engineering"
    ]

    assert fit_window_pre[0]["metadata"]["fit_panel"]["end"] == first_fit_end
    assert fit_window_features[0]["metadata"]["fit_panel"]["end"] == first_fit_end
    assert full_pre[0]["metadata"]["fit_panel"]["n_rows"] == len(panel)
    assert full_features[0]["metadata"]["fit_panel"]["n_rows"] == len(panel)
    assert [record["updated"] for record in full_pre[:2]] == [True, False]
    assert [record["updated"] for record in full_features[:2]] == [True, False]
