from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def _panel() -> pd.DataFrame:
    return mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=6, freq="MS"),
                "target": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
                "x1": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                "x2": [20.0, 22.0, 24.0, 26.0, 28.0, 30.0],
            }
        ),
        date="date",
        metadata={"dataset": "custom", "source_family": "custom", "frequency": "monthly"},
    )


def _processed() -> mf.preprocessing.PreprocessedData:
    data_spec = mf.data.spec(
        mf.data.DataBundle(_panel(), {"dataset": "custom", "source_family": "custom", "frequency": "monthly"}),
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


def _long_processed(periods: int = 36) -> mf.preprocessing.PreprocessedData:
    dates = pd.date_range("2000-01-01", periods=periods, freq="MS")
    t = np.arange(periods, dtype=float)
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": dates,
                "target": 100.0 + 0.5 * t + np.sin(t / 3.0),
                "x1": 10.0 + 0.2 * t + np.sin(t / 4.0),
                "x2": 20.0 + 0.3 * t + np.cos(t / 5.0),
            }
        ),
        date="date",
        metadata={"dataset": "custom", "source_family": "custom", "frequency": "monthly"},
    )
    data_spec = mf.data.spec(
        mf.data.DataBundle(panel, {"dataset": "custom", "source_family": "custom", "frequency": "monthly"}),
        target="target",
        horizons=[1],
        predictors=["x1", "x2"],
    )
    return mf.preprocessing.reprocess(
        data_spec,
        transform="none",
        outliers="none",
        impute="none",
        frame="keep",
    )


def _future_spiked(processed: mf.preprocessing.PreprocessedData) -> mf.preprocessing.PreprocessedData:
    panel = processed.panel.copy()
    panel.loc[panel.index[-1], ["x1", "x2"]] = [10_000.0, -5_000.0]
    return replace(processed, panel=panel)


def _assert_same_values(left: pd.DataFrame, right: pd.DataFrame) -> None:
    pd.testing.assert_index_equal(left.index, right.index)
    assert list(left.columns) == list(right.columns)
    np.testing.assert_allclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float), equal_nan=True)


def _manual_marx(
    panel: pd.DataFrame,
    *,
    columns: list[str],
    max_lag: int,
    scale_lags: bool = False,
) -> pd.DataFrame:
    lag_values = tuple(range(1, max_lag + 1))
    lag_matrix = pd.DataFrame(
        {
            f"{column}_lag{lag}": panel[column].shift(lag)
            for lag in lag_values
            for column in columns
        },
        index=panel.index,
    )
    if scale_lags:
        complete = lag_matrix.dropna()
        lag_matrix = (lag_matrix - complete.mean(axis=0)) / complete.std(axis=0, ddof=1).replace(0.0, np.nan)
    result = pd.DataFrame(index=panel.index)
    for column in columns:
        for lag in lag_values:
            lag_columns = [f"{column}_lag{step}" for step in range(1, lag + 1)]
            result[f"{column}_ma{lag}_lag1"] = lag_matrix.loc[:, lag_columns].mean(axis=1, skipna=False)
    result.index.name = "date"
    return result


def test_direct_target_builds_forecast_columns() -> None:
    target = mf.feature_engineering.direct_target(_processed(), horizons=[1, 2])

    assert list(target.columns) == ["target_level_h1", "target_level_h2"]
    assert target["target_level_h1"].iloc[:3].tolist() == [2.0, 4.0, 8.0]
    assert np.isnan(target["target_level_h2"].iloc[-2])
    target_meta = target.attrs["macroforecast_target_metadata"].set_index("target_column")
    assert target_meta.loc["target_level_h1", "mode"] == "direct"
    assert target_meta.loc["target_level_h1", "source"] == "target"
    assert target_meta.loc["target_level_h1", "horizon"] == 1
    assert target_meta.loc["target_level_h1", "formula"] == "target[t+1]"


def test_direct_target_supports_change_transforms() -> None:
    target = mf.feature_engineering.direct_target(_processed(), horizon=1, transform="change")

    assert target["target_change_h1"].iloc[:3].tolist() == [1.0, 2.0, 4.0]


def test_average_target_averages_one_period_path() -> None:
    target = mf.feature_engineering.average_target(_processed(), horizon=2, transform="change")
    same = mf.feature_engineering.direct_target(_processed(), horizon=2, transform="average_change")

    assert list(target.columns) == ["target_average_change_h2"]
    assert target["target_average_change_h2"].iloc[0] == pytest.approx(1.5)
    assert target["target_average_change_h2"].iloc[1] == pytest.approx(3.0)
    pd.testing.assert_frame_equal(target, same)
    target_meta = target.attrs["macroforecast_target_metadata"].set_index("target_column")
    assert target_meta.loc["target_average_change_h2", "operation"] == "direct_average_target"
    assert target_meta.loc["target_average_change_h2", "aggregation"] == "mean_step_change"


def test_average_target_supports_average_growth() -> None:
    target = mf.feature_engineering.average_target(_processed(), horizon=2, transform="growth")

    assert list(target.columns) == ["target_average_growth_h2"]
    assert target["target_average_growth_h2"].iloc[0] == pytest.approx(1.0)


def test_path_targets_builds_step_level_targets() -> None:
    target = mf.feature_engineering.path_targets(_processed(), horizon=3, transform="change")

    assert list(target.columns) == [
        "target_change_step1",
        "target_change_step2",
        "target_change_step3",
    ]
    assert target.iloc[0].tolist() == [1.0, 2.0, 4.0]
    stage = target.attrs["macroforecast_metadata"]["path_target"]
    assert stage["columns_by_horizon"]["h3"]["target"] == list(target.columns)
    target_meta = target.attrs["macroforecast_target_metadata"].set_index("target_column")
    assert target_meta.loc["target_change_step2", "mode"] == "path"
    assert target_meta.loc["target_change_step2", "step"] == 2
    assert target_meta.loc["target_change_step2", "aggregation"] == "average_step_forecasts_in_evaluation"
    assert target_meta.loc["target_change_step2", "used_for_horizons"] == "3"


def test_direct_target_pct_change_turns_zero_denominator_into_missing() -> None:
    panel = _panel().copy()
    panel.loc[panel.index[0], "target"] = 0.0

    target = mf.feature_engineering.direct_target(panel, target="target", horizon=1, transform="growth")

    assert np.isnan(target["target_growth_h1"].iloc[0])
    assert np.isfinite(target["target_growth_h1"].iloc[1])


def test_direct_target_metadata_covers_multiple_targets_and_horizons() -> None:
    target = mf.feature_engineering.direct_target(
        _processed(),
        targets=["target", "x1"],
        horizons=[1, 2],
        transform="change",
    )
    target_meta = target.attrs["macroforecast_target_metadata"]

    assert len(target_meta) == 4
    assert set(target_meta["source"]) == {"target", "x1"}
    assert set(target_meta["horizon"]) == {1, 2}
    assert set(target_meta["mode"]) == {"direct"}
    assert set(target_meta["operation"]) == {"direct_target"}


def test_lag_accepts_exact_lags_including_current() -> None:
    lagged = mf.feature_engineering.lag(_processed(), columns=["x1"], lags=(0, 2))

    assert list(lagged.columns) == ["x1_lag0", "x1_lag2"]
    assert lagged["x1_lag0"].iloc[0] == 10.0
    assert np.isnan(lagged["x1_lag2"].iloc[1])
    assert lagged["x1_lag2"].iloc[2] == 10.0
    feature_meta = lagged.attrs["macroforecast_feature_metadata"]
    assert set(["feature", "operation", "source", "lag", "included"]).issubset(feature_meta.columns)
    assert feature_meta.loc[feature_meta["feature"] == "x1_lag2", "lag"].iloc[0] == 2


def test_feature_metadata_schema_is_standardized() -> None:
    timed = mf.feature_engineering.time_features(_processed(), month=True)
    feature_meta = timed.attrs["macroforecast_feature_metadata"]

    assert list(feature_meta.columns[:12]) == [
        "feature",
        "step",
        "block",
        "operation",
        "source",
        "parameter",
        "lag",
        "window",
        "component",
        "fit_policy",
        "inputs",
        "included",
    ]
    assert feature_meta.attrs["macroforecast_metadata_schema"]["kind"] == "feature_metadata"
    assert feature_meta.attrs["macroforecast_metadata_schema"]["version"] == 1
    assert set(feature_meta["source"]) == {"date"}
    assert set(feature_meta["inputs"]) == {"date"}
    assert feature_meta["included"].map(type).eq(bool).all()
    assert feature_meta["included"].all()


def test_mixed_frequency_lags_align_monthly_predictors_to_quarter_end_anchor() -> None:
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=6, freq="MS"),
                "m": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "q_target": [10.0, np.nan, np.nan, 20.0, np.nan, np.nan],
            }
        ),
        date="date",
    )
    bundle = mf.data.set_frequencies(
        panel,
        {"m": "monthly", "q_target": "quarterly"},
    )

    features = mf.feature_engineering.mixed_frequency_lags(
        bundle,
        target="q_target",
        columns=["m"],
        lags=(0, 1, 2),
        target_frequency="quarterly",
        anchor_position="period_end",
    )

    assert list(features.index) == [pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-01")]
    assert list(features.columns) == ["m_lag0", "m_lag1", "m_lag2"]
    assert features.iloc[0].tolist() == [3.0, 2.0, 1.0]
    assert features.iloc[1].tolist() == [6.0, 5.0, 4.0]
    stage = features.attrs["macroforecast_metadata"]["feature_engineering_mixed_frequency_lags"]
    assert stage["frequency_by_column"] == {"m": "monthly"}
    feature_meta = features.attrs["macroforecast_feature_metadata"].set_index("feature")
    assert feature_meta.loc["m_lag2", "source_frequency"] == "monthly"
    assert feature_meta.loc["m_lag2", "lag"] == 2


def test_mixed_frequency_lags_use_native_period_not_timestamp_convention() -> None:
    idx = pd.date_range("2020-01-31", periods=6, freq="ME", name="date")
    panel = pd.DataFrame(
        {
            "m": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "q_target": [np.nan, np.nan, 10.0, np.nan, np.nan, 20.0],
        },
        index=idx,
    )
    bundle = mf.data.set_frequencies(
        panel,
        {"m": "monthly", "q_target": "quarterly"},
    )

    features = mf.feature_engineering.mixed_frequency_lags(
        bundle,
        target="q_target",
        columns=["m"],
        lags=(0, 1, 2),
        target_frequency="quarterly",
        anchor_position="period_end",
    )

    assert list(features.index) == [pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-01")]
    assert features.iloc[0].tolist() == [3.0, 2.0, 1.0]
    stage = features.attrs["macroforecast_metadata"]["feature_engineering_mixed_frequency_lags"]
    assert stage["lookup_calendar"] == "source_period_start"
    assert stage["n_rows_before_drop"] == 2
    assert stage["n_rows_after_drop"] == 2
    feature_meta = features.attrs["macroforecast_feature_metadata"].set_index("feature")
    assert feature_meta.loc["m_lag0", "lookup_start"] == "2020-03-01"


def test_rolling_mean_and_time_features_are_public_helpers() -> None:
    rolled = mf.feature_engineering.rolling_mean(_processed(), columns=["x1"], windows=3)
    timed = mf.feature_engineering.time_features(_processed(), month=True, quarter=True)

    assert rolled["x1_roll3_mean"].iloc[2] == pytest.approx(11.0)
    assert "trend" in timed.columns
    assert "month_01" in timed.columns
    assert "quarter_1" in timed.columns
    feature_meta = timed.attrs["macroforecast_feature_metadata"]
    assert set(feature_meta["source"]) == {"date"}
    assert "month=01" in set(feature_meta["parameter"])


def test_simple_transform_and_seasonal_helpers_are_public() -> None:
    panel = _processed()

    diffed = mf.feature_engineering.diff_features(panel, columns=["x1"], periods=1)
    seasonal = mf.feature_engineering.seasonal_lag(
        panel,
        columns=["x1"],
        season_length=2,
        lags=1,
    )
    dummies = mf.feature_engineering.season_dummy(panel, frequency="month", drop_first=True)

    assert diffed["x1_diff"].iloc[1] == pytest.approx(1.0)
    assert seasonal["x1_seasonlag2"].iloc[2] == pytest.approx(10.0)
    assert "month_01" not in dummies.columns
    assert dummies.attrs["macroforecast_metadata"]["feature_engineering_season_dummy"]["frequency"] == "month"
    seasonal_meta = seasonal.attrs["macroforecast_feature_metadata"].set_index("feature")
    assert seasonal_meta.loc["x1_seasonlag2", "operation"] == "seasonal_lag"
    assert seasonal_meta.loc["x1_seasonlag2", "lag"] == 2


def test_calendar_expansion_filter_and_kernel_features() -> None:
    panel = _processed()

    fourier = mf.feature_engineering.fourier_features(panel, period=4, order=2)
    poly = mf.feature_engineering.polynomial_features(panel, columns=["x1", "x2"], degree=2)
    interaction = mf.feature_engineering.interaction_features(panel, columns=["x1", "x2"], order=2)
    hp = mf.feature_engineering.hp_filter_features(
        panel,
        columns=["x1"],
        lamb=1600.0,
        warn_full_sample=False,
    )
    smooth = mf.feature_engineering.savitzky_golay_features(
        panel,
        columns=["x1"],
        window_length=5,
        warn_full_sample=False,
    )
    projected = mf.feature_engineering.random_projection_features(
        panel,
        columns=["x1", "x2"],
        n_components=1,
        random_state=0,
        warn_full_sample=False,
    )
    nystroem = mf.feature_engineering.nystroem_features(
        panel,
        columns=["x1", "x2"],
        n_components=2,
        random_state=0,
        warn_full_sample=False,
    )

    assert {"fourier_sin1_p4", "fourier_cos2_p4"}.issubset(fourier.columns)
    assert "poly_x1^2" in poly.columns
    assert list(interaction.columns) == ["interaction_x1__x2"]
    assert interaction["interaction_x1__x2"].iloc[0] == pytest.approx(
        panel.panel["x1"].iloc[0] * panel.panel["x2"].iloc[0]
    )
    assert set(interaction.attrs["macroforecast_feature_metadata"]["operation"]) == {"interaction"}
    assert "x1_hp_cycle" in hp.columns
    assert "x1_savgol" in smooth.columns
    assert list(projected.columns) == ["rp1"]
    assert list(nystroem.columns) == ["nys1", "nys2"]


def test_legacy_gap_direct_feature_transforms_are_callable() -> None:
    panel = _processed()

    wavelet = mf.feature_engineering.wavelet_features(panel, columns=["x1"], n_levels=2)
    trim = mf.feature_engineering.asymmetric_trim_features(panel, columns=["x1", "x2"])
    albama = mf.feature_engineering.adaptive_ma_rf_features(
        panel,
        columns=["x1"],
        n_estimators=3,
        min_samples_leaf=2,
        warn_full_sample=False,
    )
    pls = mf.feature_engineering.partial_least_squares_features(
        panel,
        target="target",
        columns=["x1", "x2"],
        n_components=2,
        warn_full_sample=False,
    )
    dfm = mf.feature_engineering.dfm_features(
        panel,
        columns=["x1", "x2"],
        n_factors=2,
        warn_full_sample=False,
    )
    selected = mf.feature_engineering.correlation_selection(
        panel,
        target="target",
        columns=["x1", "x2"],
        n_features=1,
        warn_full_sample=False,
    )

    assert list(wavelet.columns) == ["x1_wA1", "x1_wD1", "x1_wA2", "x1_wD2"]
    assert list(trim.columns) == ["rank_1", "rank_2"]
    assert trim.iloc[0].tolist() == [10.0, 20.0]
    assert list(albama.columns) == ["x1_albama"]
    assert list(pls.columns) == ["pls1", "pls2"]
    assert list(dfm.columns) == ["dfm1", "dfm2"]
    assert selected.shape[1] == 1
    assert dfm.attrs["macroforecast_metadata"]["feature_engineering_dfm"]["n_factors"] == 2


def test_individual_feature_selection_callables_cover_advanced_methods() -> None:
    processed = _long_processed(periods=48)
    panel = processed.panel.assign(
        x3=np.linspace(3.0, 4.0, len(processed.panel)),
        x4=np.sin(np.arange(len(processed.panel)) * 2.0),
    )
    calls = [
        mf.feature_engineering.variance_selection(
            panel,
            columns=["x1", "x2", "x3", "x4"],
            n_features=2,
        ),
        mf.feature_engineering.correlation_selection(
            panel,
            target="target",
            columns=["x1", "x2", "x3", "x4"],
            n_features=2,
            warn_full_sample=False,
        ),
        mf.feature_engineering.lasso_selection(
            panel,
            target="target",
            columns=["x1", "x2", "x3", "x4"],
            n_features=2,
            alpha=0.001,
            warn_full_sample=False,
        ),
        mf.feature_engineering.lasso_path_selection(
            panel,
            target="target",
            columns=["x1", "x2", "x3", "x4"],
            n_features=2,
            n_alphas=10,
            warn_full_sample=False,
        ),
        mf.feature_engineering.rfe_selection(
            panel,
            target="target",
            columns=["x1", "x2", "x3", "x4"],
            n_features=2,
            use_cv=False,
            warn_full_sample=False,
        ),
        mf.feature_engineering.boruta_selection(
            panel,
            target="target",
            columns=["x1", "x2", "x3", "x4"],
            n_features=2,
            n_estimators=10,
            max_iter=5,
            warn_full_sample=False,
        ),
        mf.feature_engineering.stability_selection(
            panel,
            target="target",
            columns=["x1", "x2", "x3", "x4"],
            n_features=2,
            n_subsamples=8,
            warn_full_sample=False,
        ),
        mf.feature_engineering.genetic_selection(
            panel,
            target="target",
            columns=["x1", "x2", "x3", "x4"],
            n_features=2,
            population_size=8,
            n_generations=3,
            warn_full_sample=False,
        ),
    ]

    for frame in calls:
        assert frame.shape[1] == 2
        assert frame.attrs["macroforecast_feature_metadata"]["operation"].iloc[0].endswith("_selection")


def test_hamilton_filter_features_are_expanding_by_default() -> None:
    panel = _long_processed()

    hamilton = mf.feature_engineering.hamilton_filter_features(
        panel,
        columns=["x1"],
        h=2,
        p=2,
        min_train_size=4,
        component="both",
    )

    assert list(hamilton.columns) == ["x1_hamilton_cycle", "x1_hamilton_trend"]
    assert hamilton.index.name == "date"
    assert hamilton["x1_hamilton_cycle"].notna().sum() > 0
    assert hamilton["x1_hamilton_trend"].notna().sum() > 0
    assert hamilton["x1_hamilton_cycle"].iloc[:5].isna().all()
    meta = hamilton.attrs["macroforecast_metadata"]["feature_engineering_hamilton_filter"]
    assert meta["fit_policy"] == "expanding"
    assert meta["label_alignment"] == "components are labeled at t+h"
    feature_meta = hamilton.attrs["macroforecast_feature_metadata"].set_index("feature")
    assert feature_meta.loc["x1_hamilton_cycle", "operation"] == "hamilton_filter"
    assert feature_meta.loc["x1_hamilton_cycle", "source"] == "x1"


def test_hamilton_filter_full_sample_policy_warns() -> None:
    panel = _long_processed()

    with pytest.warns(UserWarning, match="full_sample"):
        full_sample = mf.feature_engineering.hamilton_filter_features(
            panel,
            columns=["x1"],
            h=2,
            p=2,
            min_train_size=4,
            fit_policy="full_sample",
        )

    assert full_sample["x1_hamilton_cycle"].notna().sum() > 0


def test_full_input_projection_helpers_warn_by_default() -> None:
    panel = _processed()

    with pytest.warns(UserWarning, match="full_sample"):
        mf.feature_engineering.hp_filter_features(panel, columns=["x1"])
    with pytest.warns(UserWarning, match="full_sample"):
        mf.feature_engineering.savitzky_golay_features(panel, columns=["x1"], window_length=5)
    with pytest.warns(UserWarning, match="full_sample"):
        mf.feature_engineering.random_projection_features(
            panel,
            columns=["x1", "x2"],
            n_components=1,
            random_state=0,
        )
    with pytest.warns(UserWarning, match="full_sample"):
        mf.feature_engineering.nystroem_features(
            panel,
            columns=["x1", "x2"],
            n_components=1,
            random_state=0,
        )


def test_moving_average_ladder_is_ma_block_not_pca() -> None:
    ladder = mf.feature_engineering.moving_average_ladder(_processed(), columns=["x1"], max_window=5)

    assert list(ladder.columns) == ["x1_ma1", "x1_ma2", "x1_ma4"]
    assert ladder["x1_ma1"].iloc[0] == 10.0
    assert ladder["x1_ma2"].iloc[1] == pytest.approx(10.5)
    assert ladder["x1_ma4"].iloc[3] == pytest.approx(11.5)
    assert "pca" not in " ".join(ladder.columns).lower()
    assert ladder.attrs["macroforecast_metadata"]["feature_engineering_moving_average_ladder"]["windows"] == [1, 2, 4]


def test_marx_ladder_matches_author_r_loop_without_scaling() -> None:
    processed = _processed()
    matrix = mf.feature_engineering.feature_matrix(
        processed,
        specification="MARX",
        columns=["x1", "x2"],
        max_lag=3,
    )
    expected = _manual_marx(processed.panel, columns=["x1", "x2"], max_lag=3).add_prefix("MARX__")

    pd.testing.assert_frame_equal(matrix, expected)
    assert matrix["MARX__x1_ma3_lag1"].iloc[3] == pytest.approx(np.mean([12.0, 11.0, 10.0]))


def test_marx_ladder_matches_author_r_loop_with_ex_ante_scaling() -> None:
    processed = _processed()
    matrix = mf.feature_engineering.feature_matrix(
        processed,
        specification="MARX",
        columns=["x1", "x2"],
        max_lag=3,
        fit_policy="full_sample",
        min_train_size=2,
        scale_marx=True,
        warn_full_sample=False,
    )
    expected = _manual_marx(
        processed.panel,
        columns=["x1", "x2"],
        max_lag=3,
        scale_lags=True,
    ).add_prefix("MARX__")

    pd.testing.assert_frame_equal(matrix, expected)
    stage = matrix.attrs["macroforecast_metadata"]["feature_engineering_feature_matrix"]
    assert stage["scale_marx"] is True


def test_marx_step_matches_author_r_loop_without_scaling() -> None:
    processed = _processed()
    features = mf.feature_engineering.feature_spec(
        target="target",
        horizon=1,
        predictors=["x1", "x2"],
        steps=[mf.feature_engineering.marx_step(max_lag=3)],
        drop_missing=False,
    ).fit_transform(processed)
    expected = _manual_marx(processed.panel, columns=["x1", "x2"], max_lag=3)

    pd.testing.assert_frame_equal(features.X, expected)
    row = features.feature_metadata.loc[features.feature_metadata["feature"] == "x1_ma3_lag1"].iloc[0]
    assert row["operation"] == "marx"
    assert row["source"] == "x1"
    assert row["window"] == 3
    assert row["lag"] == 1


def test_feature_spec_supports_explicit_target_lags() -> None:
    processed = _processed()
    features = mf.feature_engineering.feature_spec(
        target="target",
        horizon=1,
        predictors=[],
        lags=None,
        target_lags=(0, 1),
        drop_missing=False,
    ).fit_transform(processed)

    assert list(features.X.columns) == ["target_lag0", "target_lag1"]
    assert features.metadata["feature_spec"]["spec"]["target_lags"] == [0, 1]
    row = features.feature_metadata.loc[features.feature_metadata["feature"] == "target_lag0"].iloc[0]
    assert row["operation"] == "target_lag"
    assert row["source"] == "target"


def test_marx_step_matches_author_r_loop_with_fixed_lag_scaling() -> None:
    processed = _processed()
    spec = mf.feature_engineering.feature_spec(
        target="target",
        horizon=1,
        predictors=["x1", "x2"],
        steps=[
            mf.feature_engineering.marx_step(
                max_lag=3,
                scale_lags=True,
                min_train_size=2,
            )
        ],
        drop_missing=False,
    )
    fitted = spec.fit(processed)
    features = fitted.transform(processed)
    expected = _manual_marx(processed.panel, columns=["x1", "x2"], max_lag=3, scale_lags=True)

    pd.testing.assert_frame_equal(features.X, expected)
    fit_state = fitted.to_metadata()["feature_steps"][0]["fit_state"]
    assert fit_state["scale_lags"] is True
    assert fit_state["fit_policy"] == "fixed_fit_panel"
    assert "fit_policy" not in features.metadata["feature_spec"]["spec"]["feature_steps"][0]


def test_marx_step_fixed_scaling_does_not_refit_on_transform_panel() -> None:
    processed = _processed()
    spiked = _future_spiked(processed)
    fit_processed = replace(processed, panel=processed.panel.iloc[:-1])
    spec = mf.feature_engineering.feature_spec(
        target="target",
        horizon=1,
        predictors=["x1", "x2"],
        steps=[
            mf.feature_engineering.marx_step(
                max_lag=2,
                scale_lags=True,
                min_train_size=2,
            )
        ],
        drop_missing=False,
    )

    fitted = spec.fit(fit_processed)
    base = fitted.transform(processed, index=processed.panel.index[:-1])
    changed_future = fitted.transform(spiked, index=processed.panel.index[:-1])

    _assert_same_values(base.X, changed_future.X)


def test_pca_features_defaults_to_expanding_fit_policy() -> None:
    pca = mf.feature_engineering.pca_features(
        _processed(),
        columns=["x1", "x2"],
        n_components=1,
        min_train_size=3,
    )

    assert list(pca.columns) == ["pc1"]
    assert pca.iloc[:2].isna().all().all()
    assert pca.iloc[2:].notna().all().all()
    assert pca.attrs["macroforecast_metadata"]["feature_engineering_pca"]["fit_policy"] == "expanding"


def test_full_sample_fit_policy_warns_unless_explicitly_suppressed() -> None:
    with pytest.warns(UserWarning, match="fit_policy='full_sample'"):
        mf.feature_engineering.pca_features(
            _processed(),
            columns=["x1", "x2"],
            n_components=1,
            fit_policy="full_sample",
            min_train_size=3,
        )

    quiet = mf.feature_engineering.pca_features(
        _processed(),
        columns=["x1", "x2"],
        n_components=1,
        fit_policy="full_sample",
        min_train_size=3,
        warn_full_sample=False,
    )

    assert list(quiet.columns) == ["pc1"]


def test_group_pca_builds_factors_within_named_groups() -> None:
    grouped = mf.feature_engineering.group_pca(
        _processed(),
        groups={"slow": ["x1", "x2"], "fast": ["x1"]},
        n_components={"slow": 1, "fast": 1},
        fit_policy="full_sample",
        min_train_size=3,
        scale=True,
        warn_full_sample=False,
    )
    expected_slow = mf.feature_engineering.pca_features(
        _processed(),
        columns=["x1", "x2"],
        n_components=1,
        fit_policy="full_sample",
        min_train_size=3,
        scale=True,
        prefix="slow",
        warn_full_sample=False,
    )

    assert list(grouped.columns) == ["slow1", "fast1"]
    pd.testing.assert_frame_equal(grouped.loc[:, ["slow1"]], expected_slow)
    stage = grouped.attrs["macroforecast_metadata"]["feature_engineering_group_pca"]
    assert [group["group"] for group in stage["groups"]] == ["slow", "fast"]


def test_expanding_fit_policy_does_not_use_future_rows() -> None:
    processed = _processed()
    spiked = _future_spiked(processed)

    scale_base = mf.feature_engineering.scale_features(
        processed,
        columns=["x1", "x2"],
        fit_policy="expanding",
        min_train_size=3,
    )
    scale_spiked = mf.feature_engineering.scale_features(
        spiked,
        columns=["x1", "x2"],
        fit_policy="expanding",
        min_train_size=3,
    )
    pca_base = mf.feature_engineering.pca_features(
        processed,
        columns=["x1", "x2"],
        n_components=1,
        fit_policy="expanding",
        min_train_size=3,
    )
    pca_spiked = mf.feature_engineering.pca_features(
        spiked,
        columns=["x1", "x2"],
        n_components=1,
        fit_policy="expanding",
        min_train_size=3,
    )
    group_base = mf.feature_engineering.group_pca(
        processed,
        groups={"both": ["x1", "x2"]},
        n_components=1,
        fit_policy="expanding",
        min_train_size=3,
    )
    group_spiked = mf.feature_engineering.group_pca(
        spiked,
        groups={"both": ["x1", "x2"]},
        n_components=1,
        fit_policy="expanding",
        min_train_size=3,
    )
    maf_base = mf.feature_engineering.maf_features(
        processed,
        columns=["x1", "x2"],
        max_lag=1,
        n_components=1,
        fit_policy="expanding",
        min_train_size=3,
    )
    maf_spiked = mf.feature_engineering.maf_features(
        spiked,
        columns=["x1", "x2"],
        max_lag=1,
        n_components=1,
        fit_policy="expanding",
        min_train_size=3,
    )

    _assert_same_values(scale_base.iloc[:-1], scale_spiked.iloc[:-1])
    _assert_same_values(pca_base.iloc[:-1], pca_spiked.iloc[:-1])
    _assert_same_values(group_base.iloc[:-1], group_spiked.iloc[:-1])
    _assert_same_values(maf_base.iloc[:-1], maf_spiked.iloc[:-1])


def test_compose_features_supports_group_pca_step() -> None:
    grouped = mf.feature_engineering.compose_features(
        _processed(),
        [
            mf.feature_engineering.group_pca_step(
                groups={"slow": ["x1", "x2"]},
                n_components=1,
                fit_policy="full_sample",
                min_train_size=3,
                warn_full_sample=False,
            )
        ],
    )

    assert list(grouped.columns) == ["slow1"]


def test_sparse_pca_chen_rohe_features_full_sample_callable() -> None:
    sparse = mf.feature_engineering.sparse_pca_chen_rohe_features(
        _long_processed(),
        columns=["x1", "x2"],
        n_components=2,
        max_iter=40,
        var_innovations=True,
        prefix="scaf",
        warn_full_sample=False,
    )

    assert list(sparse.columns) == ["scaf1", "scaf2"]
    assert sparse.notna().all().all()
    stage = sparse.attrs["macroforecast_metadata"]["feature_engineering_sparse_pca_chen_rohe"]
    assert stage["resolved_n_components"] == 2
    assert stage["var_innovations"] is True
    assert stage["fit_policy"] == "full_input_complete_rows"
    feature_meta = sparse.attrs["macroforecast_feature_metadata"]
    assert feature_meta.loc[0, "operation"] == "sparse_pca_chen_rohe"


def test_compose_features_supports_sparse_pca_chen_rohe_step() -> None:
    sparse = mf.feature_engineering.compose_features(
        _long_processed(),
        [
            mf.feature_engineering.sparse_pca_chen_rohe_step(
                n_components=1,
                max_iter=30,
                min_train_size=12,
                warn_full_sample=False,
            )
        ],
    )

    assert list(sparse.columns) == ["sca1"]


def test_varimax_features_rotates_factor_scores() -> None:
    factors = mf.feature_engineering.pca_features(
        _long_processed(),
        columns=["x1", "x2"],
        n_components=2,
        fit_policy="full_sample",
        min_train_size=12,
        warn_full_sample=False,
    )
    rotated = mf.feature_engineering.varimax_features(
        factors,
        max_iter=20,
        warn_full_sample=False,
    )

    assert list(rotated.columns) == ["varimax1", "varimax2"]
    assert rotated.notna().all().all()
    stage = rotated.attrs["macroforecast_metadata"]["feature_engineering_varimax"]
    assert stage["fit_policy"] == "full_input_complete_rows"
    assert rotated.attrs["macroforecast_feature_metadata"].loc[0, "operation"] == "varimax"


def test_sliced_inverse_regression_features_are_target_aware_direct_callable() -> None:
    processed = _long_processed()
    sir = mf.feature_engineering.sliced_inverse_regression_features(
        processed,
        target="target",
        columns=["x1", "x2"],
        n_components=2,
        n_slices=4,
        scaling_policy="scaled_pca",
        warn_full_sample=False,
    )

    assert list(sir.columns) == ["sir1", "sir2"]
    assert sir.notna().all().all()
    stage = sir.attrs["macroforecast_metadata"]["feature_engineering_sliced_inverse_regression"]
    assert stage["target"] == "target"
    assert stage["fit_policy"] == "full_input_target_aligned_rows"
    assert sir.attrs["macroforecast_feature_metadata"].loc[0, "operation"] == "sliced_inverse_regression"


def test_compose_features_supports_varimax_step() -> None:
    rotated = mf.feature_engineering.compose_features(
        _long_processed(),
        [
            mf.feature_engineering.pca_step(
                name="pc",
                n_components=2,
                fit_policy="full_sample",
                min_train_size=12,
                include=False,
                warn_full_sample=False,
            ),
            mf.feature_engineering.varimax_step(
                name="rot",
                input="pc",
                max_iter=20,
                warn_full_sample=False,
            ),
        ],
    )

    assert list(rotated.columns) == ["rot1", "rot2"]


def test_maf_features_runs_pca_within_each_variable_lag_panel() -> None:
    maf = mf.feature_engineering.maf_features(
        _processed(),
        columns=["x1", "x2"],
        max_lag=2,
        n_components=2,
        fit_policy="full_sample",
        min_train_size=3,
        warn_full_sample=False,
    )

    assert list(maf.columns) == ["x1_maf1", "x1_maf2", "x2_maf1", "x2_maf2"]
    assert maf.iloc[:2].isna().all().all()
    assert maf.iloc[2:].notna().all().all()
    stage = maf.attrs["macroforecast_metadata"]["feature_engineering_maf"]
    assert stage["lags"] == [0, 1, 2]
    assert "global PCA" in stage["note"]
    feature_meta = maf.attrs["macroforecast_feature_metadata"]
    assert set(feature_meta["source"]) == {"x1", "x2"}

    x1_lag_panel = mf.feature_engineering.lag(_processed(), columns=["x1"], lags=[0, 1, 2])
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


def test_compose_features_supports_pca_then_lag_and_lag_then_pca() -> None:
    pca_then_lag = mf.feature_engineering.compose_features(
        _processed(),
        [
            mf.feature_engineering.pca_step(
                name="pc",
                columns=["x1", "x2"],
                n_components=1,
                min_train_size=3,
                include=False,
            ),
            mf.feature_engineering.lag_step(name="lag_pc", input="pc", lags=[1, 2]),
        ],
    )
    lag_then_pca = mf.feature_engineering.compose_features(
        _processed(),
        [
            mf.feature_engineering.lag_step(name="x_lag", columns=["x1", "x2"], lags=[0, 1], include=False),
            mf.feature_engineering.pca_step(name="lag_pc", input="x_lag", n_components=1, min_train_size=4),
        ],
    )

    assert list(pca_then_lag.columns) == ["pc1_lag1", "pc1_lag2"]
    assert list(lag_then_pca.columns) == ["lag_pc1"]


def test_compose_features_supports_deterministic_transform_steps() -> None:
    composed = mf.feature_engineering.compose_features(
        _processed(),
        [
            mf.feature_engineering.transform_step(
                name="log_x1",
                transform="log",
                columns=["x1"],
                include=False,
            ),
            mf.feature_engineering.lag_step(name="log_x1_lag", input="log_x1", lags=[1]),
            mf.feature_engineering.polynomial_step(
                name="poly",
                columns=["x1", "x2"],
                degree=2,
                include=False,
            ),
            mf.feature_engineering.interaction_step(name="cross", columns=["x1", "x2"]),
        ],
    )

    assert list(composed.columns) == ["x1_log_lag1", "interaction_x1__x2"]
    assert composed["x1_log_lag1"].iloc[1] == pytest.approx(np.log(10.0))
    assert composed["interaction_x1__x2"].iloc[0] == pytest.approx(200.0)
    feature_meta = composed.attrs["macroforecast_feature_metadata"].set_index("feature")
    assert feature_meta.loc["x1_log", "step"] == "log_x1"
    assert not bool(feature_meta.loc["poly_x1^2", "included"])
    assert feature_meta.loc["interaction_x1__x2", "operation"] == "interaction"


def test_compose_features_supports_hamilton_step() -> None:
    composed = mf.feature_engineering.compose_features(
        _long_processed(),
        [
            mf.feature_engineering.hamilton_step(
                name="hamilton",
                columns=["x1"],
                h=2,
                p=2,
                min_train_size=4,
            ),
        ],
    )

    assert list(composed.columns) == ["x1_hamilton_cycle"]
    assert composed["x1_hamilton_cycle"].notna().sum() > 0
    feature_meta = composed.attrs["macroforecast_feature_metadata"].set_index("feature")
    assert feature_meta.loc["x1_hamilton_cycle", "step"] == "hamilton"
    assert feature_meta.loc["x1_hamilton_cycle", "operation"] == "hamilton_filter"


def test_compose_features_supports_projection_steps() -> None:
    composed = mf.feature_engineering.compose_features(
        _long_processed(),
        [
            mf.feature_engineering.random_projection_step(
                name="rp",
                columns=["x1", "x2"],
                n_components=2,
                random_state=0,
                warn_full_sample=False,
            ),
            mf.feature_engineering.nystroem_step(
                name="nys",
                columns=["x1", "x2"],
                n_components=2,
                random_state=0,
                warn_full_sample=False,
            ),
        ],
    )

    assert list(composed.columns) == ["rp1", "rp2", "nys1", "nys2"]
    assert composed.notna().all(axis=None)
    feature_meta = composed.attrs["macroforecast_feature_metadata"].set_index("feature")
    assert feature_meta.loc["rp1", "step"] == "rp"
    assert feature_meta.loc["rp1", "operation"] == "random_projection"
    assert feature_meta.loc["nys1", "step"] == "nys"
    assert feature_meta.loc["nys1", "operation"] == "nystroem"


def test_feature_spec_supports_runner_safe_hamilton_step() -> None:
    processed = _long_processed()
    spec = mf.feature_engineering.feature_spec(
        target="target",
        horizon=1,
        predictors=["x1", "x2"],
        steps=[
            mf.feature_engineering.hamilton_step(
                columns=["x1"],
                h=2,
                p=2,
                min_train_size=4,
                component="both",
                fit_policy="full_sample",
            )
        ],
        drop_missing=False,
    )

    fitted = spec.fit(processed)
    features = fitted.transform(processed)

    assert list(features.X.columns) == ["x1_hamilton_cycle", "x1_hamilton_trend"]
    assert features.X["x1_hamilton_cycle"].notna().sum() > 0
    step_meta = fitted.to_metadata()["feature_steps"][0]
    assert step_meta["method"] == "hamilton_filter"
    assert step_meta["params"]["fit_policy"] == "fixed_fit_panel"
    assert step_meta["fit_state"]["fit_policy"] == "fixed_fit_panel"
    assert step_meta["fit_state"]["fit_rows_by_column"]["x1"] >= 4
    assert "fit_policy" not in spec.to_dict()["feature_steps"][0]
    feature_meta = features.feature_metadata.set_index("feature")
    assert feature_meta.loc["x1_hamilton_cycle", "operation"] == "hamilton_filter"
    assert feature_meta.loc["x1_hamilton_cycle", "fit_policy"] == "fixed_fit_panel"
    assert feature_meta.loc["x1_hamilton_cycle", "h"] == 2
    assert feature_meta.loc["x1_hamilton_cycle", "p"] == 2


def test_feature_spec_hamilton_step_rejects_interpolate_missing_policy() -> None:
    spec = mf.feature_engineering.feature_spec(
        target="target",
        horizon=1,
        predictors=["x1", "x2"],
        steps=[mf.feature_engineering.hamilton_step(columns=["x1"], missing="interpolate")],
    )

    with pytest.raises(ValueError, match="missing='drop'"):
        spec.fit(_long_processed())


def test_feature_spec_supports_runner_safe_projection_steps() -> None:
    processed = _long_processed()
    spec = mf.feature_engineering.feature_spec(
        target="target",
        horizon=1,
        predictors=["x1", "x2"],
        steps=[
            mf.feature_engineering.random_projection_step(
                name="rp",
                n_components=2,
                random_state=0,
                min_train_size=5,
            ),
            mf.feature_engineering.nystroem_step(
                name="nys",
                n_components=2,
                random_state=0,
                min_train_size=5,
            ),
        ],
        drop_missing=False,
    )

    fitted = spec.fit(processed)
    features = fitted.transform(processed)

    assert list(features.X.columns) == ["rp1", "rp2", "nys1", "nys2"]
    step_meta = fitted.to_metadata()["feature_steps"]
    assert step_meta[0]["method"] == "random_projection"
    assert step_meta[0]["fit_state"]["fit_policy"] == "fixed_fit_panel"
    assert step_meta[0]["fit_state"]["n_fit_rows"] >= 5
    assert step_meta[1]["method"] == "nystroem"
    assert step_meta[1]["fit_state"]["fit_policy"] == "fixed_fit_panel"
    feature_meta = features.feature_metadata.set_index("feature")
    assert feature_meta.loc["rp1", "fit_policy"] == "fixed_fit_panel"
    assert feature_meta.loc["nys1", "fit_policy"] == "fixed_fit_panel"


def test_feature_spec_supports_runner_safe_target_aware_steps() -> None:
    processed = _long_processed()
    spec = mf.feature_engineering.feature_spec(
        target="target",
        horizon=1,
        predictors=["x1", "x2"],
        steps=[
            mf.feature_engineering.scale_step(
                name="scaled",
                columns=["x1", "x2"],
                include=False,
            ),
            mf.feature_engineering.partial_least_squares_step(
                name="pls",
                input="scaled",
                n_components=2,
                min_train_size=5,
            ),
            mf.feature_engineering.sliced_inverse_regression_step(
                name="sir",
                input="scaled",
                n_components=1,
                n_slices=4,
                min_train_size=5,
            ),
            {
                "name": "picked",
                "method": "correlation_selection",
                "input": "scaled",
                "n_features": 1,
                "min_train_size": 5,
            },
        ],
        drop_missing=False,
    )

    fitted = spec.fit(processed)
    features = fitted.transform(processed)

    assert {"pls1", "pls2", "sir1"}.issubset(features.X.columns)
    assert features.X.shape[1] == 4
    step_meta = fitted.to_metadata()["feature_steps"]
    assert step_meta[1]["method"] == "partial_least_squares"
    assert step_meta[1]["fit_state"]["target"] == "target_level_h1"
    assert step_meta[1]["fit_state"]["fit_policy"] == "fixed_fit_panel_target_aligned_rows"
    assert step_meta[2]["method"] == "sliced_inverse_regression"
    assert step_meta[3]["fit_state"]["method"] == "correlation_selection"
    feature_meta = features.feature_metadata.set_index("feature")
    assert feature_meta.loc["pls1", "operation"] == "partial_least_squares"
    assert feature_meta.loc["sir1", "operation"] == "sliced_inverse_regression"
    picked_feature = step_meta[3]["fit_state"]["selected_columns"][0]
    picked_rows = feature_meta.loc[[picked_feature]]
    assert "correlation_selection" in set(picked_rows["operation"])


def test_target_aware_feature_spec_steps_do_not_refit_on_transform_panel() -> None:
    processed = _long_processed()
    spiked = _future_spiked(processed)
    fit_processed = replace(processed, panel=processed.panel.iloc[:-2])
    spec = mf.feature_engineering.feature_spec(
        target="target",
        horizon=1,
        predictors=["x1", "x2"],
        steps=[
            mf.feature_engineering.partial_least_squares_step(
                name="pls",
                n_components=1,
                min_train_size=5,
            )
        ],
        drop_missing=False,
    )

    fitted = spec.fit(fit_processed)
    index = processed.panel.index[:-2]
    base = fitted.transform(processed, index=index)
    changed_future = fitted.transform(spiked, index=index)

    _assert_same_values(base.X, changed_future.X)


def test_feature_spec_supports_individual_selection_methods() -> None:
    processed = _long_processed(periods=48)
    methods = [
        ("variance_selection", {}),
        ("correlation_selection", {}),
        ("lasso_selection", {"alpha": 0.001}),
        ("lasso_path_selection", {"n_alphas": 10}),
        ("rfe_selection", {"use_cv": False}),
        ("boruta_selection", {"n_estimators": 10, "max_iter": 5}),
        ("stability_selection", {"n_subsamples": 8}),
        ("genetic_selection", {"population_size": 8, "n_generations": 3}),
    ]

    for method, params in methods:
        spec = mf.feature_engineering.feature_spec(
            target="target",
            horizon=1,
            predictors=["x1", "x2"],
            steps=[
                {
                    "name": method,
                    "method": method,
                    "n_features": 1,
                    "min_train_size": 5,
                    "random_state": 0,
                    **params,
                }
            ],
            drop_missing=False,
        )
        fitted = spec.fit(processed)
        features = fitted.transform(processed)
        step_meta = fitted.to_metadata()["feature_steps"][0]
        assert features.X.shape[1] == 1
        assert step_meta["method"] == method
        assert step_meta["fit_state"]["method"] == method
        assert step_meta["fit_state"]["fit_policy"].startswith("fixed_fit_panel")
        assert features.feature_metadata["operation"].iloc[0] == method


def test_target_aware_feature_spec_steps_require_single_resolved_target() -> None:
    spec = mf.feature_engineering.feature_spec(
        targets=["target", "x1"],
        horizon=1,
        predictors=["x2"],
        steps=[mf.feature_engineering.partial_least_squares_step(n_components=1)],
    )

    with pytest.raises(ValueError, match="exactly one resolved target"):
        spec.fit(_long_processed())


def test_feature_spec_supports_fit_aware_step_pipeline() -> None:
    processed = _processed()
    spec = mf.feature_engineering.feature_spec(
        target="target",
        horizon=1,
        predictors=["x1", "x2"],
        steps=[
            mf.feature_engineering.scale_step(name="scaled", columns=["x1", "x2"], include=False),
            mf.feature_engineering.pca_step(
                name="pc",
                input="scaled",
                n_components=1,
                min_train_size=3,
                include=False,
            ),
            mf.feature_engineering.lag_step(name="pc_lag", input="pc", lags=[0, 1]),
        ],
    )

    fitted = spec.fit(processed)
    features = fitted.transform(processed)

    assert list(features.X.columns) == ["pc1_lag0", "pc1_lag1"]
    assert features.metadata["feature_spec"]["spec"]["feature_steps"][0]["method"] == "scale"
    assert "fit_policy" not in features.metadata["feature_spec"]["spec"]["feature_steps"][0]
    assert fitted.to_metadata()["feature_steps"][0]["fit_state"]["method"] == "zscore"
    feature_meta = features.feature_metadata.set_index("feature")
    assert not bool(feature_meta.loc["x1_zscore", "included"])
    assert feature_meta.loc["pc1", "fit_policy"] == "fixed_fit_panel"
    assert feature_meta.loc["pc1_lag1", "step"] == "pc_lag"


def test_feature_spec_supports_runner_safe_deterministic_steps() -> None:
    processed = _processed()
    spec = mf.feature_engineering.feature_spec(
        target="target",
        horizon=1,
        predictors=["x1", "x2"],
        steps=[
            mf.feature_engineering.transform_step(
                name="x1_growth",
                transform="pct_change",
                columns=["x1"],
                include=False,
            ),
            mf.feature_engineering.lag_step(name="x1_growth_lag", input="x1_growth", lags=[1]),
            mf.feature_engineering.seasonal_lag_step(
                name="seasonal",
                columns=["x1"],
                season_length=2,
                lags=[1],
            ),
            mf.feature_engineering.season_dummy_step(name="quarter", frequency="quarter", drop_first=True),
            mf.feature_engineering.fourier_step(name="fourier", period=6, order=1, prefix="f"),
            mf.feature_engineering.time_step(name="time", trend=True, month=False, quarter=False, year=False),
            mf.feature_engineering.polynomial_step(
                name="poly",
                columns=["x1", "x2"],
                degree=2,
                include=False,
            ),
            mf.feature_engineering.interaction_step(name="cross", columns=["x1", "x2"]),
        ],
        drop_missing=False,
    )

    features = spec.fit_transform(processed)

    expected_columns = {
        "x1_pct_change_lag1",
        "x1_seasonlag2",
        "quarter_2",
        "quarter_3",
        "quarter_4",
        "f_sin1_p6",
        "f_cos1_p6",
        "trend",
        "interaction_x1__x2",
    }
    assert expected_columns.issubset(set(features.X.columns))
    assert features.X["x1_pct_change_lag1"].iloc[2] == pytest.approx((11.0 / 10.0) - 1.0)
    assert features.X["x1_seasonlag2"].iloc[2] == pytest.approx(10.0)
    feature_meta = features.feature_metadata.set_index("feature")
    assert feature_meta.loc["x1_pct_change", "operation"] == "pct_change"
    assert feature_meta.loc["x1_pct_change_lag1", "step"] == "x1_growth_lag"
    assert feature_meta.loc["quarter_2", "operation"] == "season_dummy"
    assert feature_meta.loc["f_sin1_p6", "operation"] == "fourier"
    assert feature_meta.loc["trend", "operation"] == "time"
    assert not bool(feature_meta.loc["poly_x1^2", "included"])
    assert features.metadata["feature_spec"]["spec"]["feature_steps"][0]["method"] == "transform"


def test_feature_spec_fit_aware_steps_do_not_refit_on_transform_panel() -> None:
    processed = _processed()
    spiked = _future_spiked(processed)
    spec = mf.feature_engineering.feature_spec(
        target="target",
        horizon=1,
        predictors=["x1", "x2"],
        steps=[
            mf.feature_engineering.scale_step(name="scaled", include=False),
            mf.feature_engineering.pca_step(
                name="pc",
                input="scaled",
                n_components=1,
                min_train_size=3,
            ),
        ],
    )

    fitted = spec.fit(replace(processed, panel=processed.panel.iloc[:-1]))
    base = fitted.transform(processed, index=processed.panel.index[:-1])
    changed_future = fitted.transform(spiked, index=processed.panel.index[:-1])

    _assert_same_values(base.X, changed_future.X)


def test_feature_spec_sparse_pca_chen_rohe_step_is_fit_aware() -> None:
    processed = _long_processed()
    spiked = _future_spiked(processed)
    spec = mf.feature_engineering.feature_spec(
        target="target",
        horizon=1,
        predictors=["x1", "x2"],
        steps=[
            mf.feature_engineering.sparse_pca_chen_rohe_step(
                name="sca",
                n_components=1,
                max_iter=30,
                min_train_size=12,
            )
        ],
    )

    fitted = spec.fit(replace(processed, panel=processed.panel.iloc[:-1]))
    base = fitted.transform(processed, index=processed.panel.index[:-1])
    changed_future = fitted.transform(spiked, index=processed.panel.index[:-1])

    _assert_same_values(base.X, changed_future.X)
    assert list(base.X.columns) == ["sca1"]
    assert base.feature_metadata.loc[0, "operation"] == "sparse_pca_chen_rohe"
    assert fitted.to_metadata()["feature_steps"][0]["fit_state"]["fit_policy"] == "fixed_fit_panel"


def test_feature_spec_varimax_step_is_fit_aware() -> None:
    processed = _long_processed()
    spiked = _future_spiked(processed)
    spec = mf.feature_engineering.feature_spec(
        target="target",
        horizon=1,
        predictors=["x1", "x2"],
        steps=[
            mf.feature_engineering.pca_step(
                name="pc",
                n_components=2,
                min_train_size=12,
                include=False,
            ),
            mf.feature_engineering.varimax_step(
                name="rot",
                input="pc",
                max_iter=20,
            ),
        ],
    )

    fitted = spec.fit(replace(processed, panel=processed.panel.iloc[:-1]))
    base = fitted.transform(processed, index=processed.panel.index[:-1])
    changed_future = fitted.transform(spiked, index=processed.panel.index[:-1])

    _assert_same_values(base.X, changed_future.X)
    assert list(base.X.columns) == ["rot1", "rot2"]
    assert set(base.feature_metadata["operation"]) == {"pca", "varimax"}
    assert fitted.to_metadata()["feature_steps"][1]["fit_state"]["fit_policy"] == "fixed_fit_panel"


def test_feature_spec_supports_group_pca_and_maf_steps() -> None:
    processed = _processed()
    grouped = mf.feature_engineering.feature_spec(
        target="target",
        horizon=1,
        predictors=["x1", "x2"],
        steps=[
            mf.feature_engineering.group_pca_step(
                name="groups",
                groups={"both": ["x1", "x2"]},
                n_components=1,
                min_train_size=3,
            )
        ],
    ).fit_transform(processed)
    maf = mf.feature_engineering.feature_spec(
        target="target",
        horizon=1,
        predictors=["x1", "x2"],
        steps=[
            mf.feature_engineering.maf_step(
                name="maf",
                columns=["x1"],
                max_lag=1,
                n_components=1,
                min_train_size=2,
            )
        ],
    ).fit_transform(processed)

    assert list(grouped.X.columns) == ["both1"]
    assert list(maf.X.columns) == ["x1_maf1"]
    assert grouped.feature_metadata.loc[0, "operation"] == "group_pca"
    assert maf.feature_metadata.loc[0, "operation"] == "maf"


def test_composed_feature_convenience_callables_are_short() -> None:
    pca_lags = mf.feature_engineering.pca_then_lags(
        _processed(),
        columns=["x1", "x2"],
        n_components=1,
        min_train_size=3,
        lags=[1, 2],
    )
    lag_pca = mf.feature_engineering.lags_then_pca(
        _processed(),
        columns=["x1", "x2"],
        lags=[0, 1],
        n_components=1,
        min_train_size=4,
    )
    ma_pca_lags = mf.feature_engineering.moving_average_pca_lags(
        _processed(),
        columns=["x1"],
        windows=[1, 2],
        n_components=1,
        min_train_size=2,
        lags=[1],
    )

    assert list(pca_lags.columns) == ["pc1", "pc1_lag1", "pc1_lag2"]
    assert list(lag_pca.columns) == ["lag_pc1"]
    assert list(ma_pca_lags.columns) == ["ma_pc1", "ma_pc1_lag1"]


def test_compose_features_supports_maf_step() -> None:
    maf = mf.feature_engineering.compose_features(
        _processed(),
        [
            mf.feature_engineering.maf_step(
                name="maf",
                columns=["x1"],
                max_lag=2,
                n_components=1,
                fit_policy="full_sample",
                min_train_size=3,
                warn_full_sample=False,
            )
        ],
    )

    assert list(maf.columns) == ["x1_maf1"]


def test_feature_matrix_builds_paper_style_blocks() -> None:
    matrix = mf.feature_engineering.feature_matrix(
        _processed(),
        specification="F-X-MARX",
        columns=["x1", "x2"],
        lags=[0, 1],
        max_lag=2,
        n_factors=1,
        fit_policy="full_sample",
        min_train_size=3,
        drop_missing=True,
        warn_full_sample=False,
    )

    assert "F__F1_lag0" in matrix.columns
    assert "X__x1_lag1" in matrix.columns
    assert "MARX__x1_ma1_lag1" in matrix.columns
    stage = matrix.attrs["macroforecast_metadata"]["feature_engineering_feature_matrix"]
    assert stage["specification"] == "F-X-MARX"
    assert [block["block"] for block in stage["blocks"]] == ["F", "X", "MARX"]
    feature_meta = matrix.attrs["macroforecast_feature_metadata"]
    marx_row = feature_meta.loc[feature_meta["feature"] == "MARX__x1_ma2_lag1"].iloc[0]
    assert marx_row["block"] == "MARX"
    assert marx_row["operation"] == "marx"
    assert marx_row["source"] == "x1"
    assert marx_row["window"] == 2
    assert marx_row["lag"] == 1


def test_feature_matrix_full_sample_warning_is_not_duplicated() -> None:
    with pytest.warns(UserWarning) as warnings:
        mf.feature_engineering.feature_matrix(
            _processed(),
            specification="F-X-MAF",
            columns=["x1", "x2"],
            lags=[0, 1],
            max_lag=2,
            n_factors=1,
            n_maf_components=1,
            fit_policy="full_sample",
            min_train_size=3,
        )

    full_sample_warnings = [
        item for item in warnings if "fit_policy='full_sample'" in str(item.message)
    ]
    assert len(full_sample_warnings) == 1


def test_feature_matrix_supports_paper_specification_family() -> None:
    processed = _processed()
    level_panel = _panel()

    fx = mf.feature_engineering.feature_matrix(
        processed,
        specification="F-X",
        columns=["x1", "x2"],
        lags=[0, 1],
        n_factors=1,
        fit_policy="full_sample",
        min_train_size=3,
        warn_full_sample=False,
    )
    fxh = mf.feature_engineering.feature_matrix(
        processed,
        specification="F-X-H",
        columns=["x1", "x2"],
        level_data=level_panel,
        lags=[0, 1],
        n_factors=1,
        fit_policy="full_sample",
        min_train_size=3,
        warn_full_sample=False,
    )
    fxmaf = mf.feature_engineering.feature_matrix(
        processed,
        specification="F_X_MAF",
        columns=["x1", "x2"],
        lags=[0, 1],
        max_lag=2,
        n_factors=1,
        n_maf_components=1,
        fit_policy="full_sample",
        min_train_size=3,
        warn_full_sample=False,
    )
    fxmarx = mf.feature_engineering.feature_matrix(
        processed,
        specification="F+X+MARX",
        columns=["x1", "x2"],
        lags=[0, 1],
        max_lag=2,
        n_factors=1,
        fit_policy="full_sample",
        min_train_size=3,
        warn_full_sample=False,
    )

    assert fx.attrs["macroforecast_metadata"]["feature_engineering_feature_matrix"]["specification"] == "F-X"
    assert "LEVEL__x1_lag0" in fxh.columns
    assert fxh.attrs["macroforecast_metadata"]["feature_engineering_feature_matrix"]["specification"] == "F-X-LEVEL"
    assert "MAF__x1_maf1" in fxmaf.columns
    assert fxmaf.attrs["macroforecast_metadata"]["feature_engineering_feature_matrix"]["specification"] == "F-X-MAF"
    assert "MARX__x1_ma2_lag1" in fxmarx.columns
    assert fxmarx.attrs["macroforecast_metadata"]["feature_engineering_feature_matrix"]["specification"] == "F-X-MARX"


def test_feature_matrix_supports_level_and_extended_paper_specs() -> None:
    processed = _processed()
    level_panel = _panel()

    h = mf.feature_engineering.feature_matrix(
        processed,
        specification="H",
        columns=["x1", "x2"],
        level_data=level_panel,
        lags=[0, 1],
    )
    xh = mf.feature_engineering.feature_matrix(
        processed,
        specification="X-H",
        columns=["x1", "x2"],
        level_data=level_panel,
        lags=[0, 1],
    )
    fxhmarx = mf.feature_engineering.feature_matrix(
        processed,
        specification="F-X-H-MARX",
        columns=["x1", "x2"],
        level_data=level_panel,
        lags=[0, 1],
        max_lag=2,
        n_factors=1,
        fit_policy="full_sample",
        min_train_size=3,
        warn_full_sample=False,
    )
    fxhmaf = mf.feature_engineering.feature_matrix(
        processed,
        specification="F-X-H-MAF",
        columns=["x1", "x2"],
        level_data=level_panel,
        lags=[0, 1],
        max_lag=2,
        n_factors=1,
        n_maf_components=1,
        fit_policy="full_sample",
        min_train_size=3,
        warn_full_sample=False,
    )

    assert list(h.columns) == ["LEVEL__x1_lag0", "LEVEL__x1_lag1", "LEVEL__x2_lag0", "LEVEL__x2_lag1"]
    assert "X__x1_lag0" in xh.columns
    assert "LEVEL__x1_lag0" in xh.columns
    assert fxhmarx.attrs["macroforecast_metadata"]["feature_engineering_feature_matrix"]["specification"] == "F-X-LEVEL-MARX"
    assert "LEVEL__x1_lag0" in fxhmarx.columns
    assert "MARX__x1_ma2_lag1" in fxhmarx.columns
    assert fxhmaf.attrs["macroforecast_metadata"]["feature_engineering_feature_matrix"]["specification"] == "F-X-LEVEL-MAF"
    assert "LEVEL__x1_lag0" in fxhmaf.columns
    assert "MAF__x1_maf1" in fxhmaf.columns


def test_feature_metadata_records_all_paper_blocks() -> None:
    matrix = mf.feature_engineering.feature_matrix(
        _processed(),
        specification="F-X-H-MAF",
        columns=["x1", "x2"],
        level_data=_panel(),
        lags=[0, 1],
        max_lag=2,
        n_factors=1,
        n_maf_components=1,
        fit_policy="full_sample",
        min_train_size=3,
        warn_full_sample=False,
    )
    feature_meta = matrix.attrs["macroforecast_feature_metadata"].set_index("feature")

    assert feature_meta.loc["F__F1_lag0", "block"] == "F"
    assert feature_meta.loc["F__F1_lag0", "operation"] == "factor_lag"
    assert feature_meta.loc["F__F1_lag0", "component"] == 1
    assert feature_meta.loc["F__F1_lag0", "fit_policy"] == "full_sample"
    assert feature_meta.loc["X__x1_lag1", "block"] == "X"
    assert feature_meta.loc["X__x1_lag1", "source"] == "x1"
    assert feature_meta.loc["X__x1_lag1", "lag"] == 1
    assert feature_meta.loc["LEVEL__x1_lag0", "block"] == "LEVEL"
    assert feature_meta.loc["LEVEL__x1_lag0", "operation"] == "level_lag"
    assert feature_meta.loc["MAF__x1_maf1", "block"] == "MAF"
    assert feature_meta.loc["MAF__x1_maf1", "source"] == "x1"
    assert feature_meta.loc["MAF__x1_maf1", "component"] == 1


def test_compose_feature_metadata_records_step_inclusion() -> None:
    features = mf.feature_engineering.compose_features(
        _processed(),
        [
            mf.feature_engineering.pca_step(
                name="pc",
                columns=["x1", "x2"],
                n_components=1,
                fit_policy="full_sample",
                min_train_size=3,
                include=False,
                warn_full_sample=False,
            ),
            mf.feature_engineering.lag_step(name="pc_lag", input="pc", lags=[1]),
        ],
    )
    feature_meta = features.attrs["macroforecast_feature_metadata"]

    pc_row = feature_meta.loc[feature_meta["feature"] == "pc1"].iloc[0]
    lag_row = feature_meta.loc[feature_meta["feature"] == "pc1_lag1"].iloc[0]
    assert pc_row["step"] == "pc"
    assert bool(pc_row["included"]) is False
    assert lag_row["step"] == "pc_lag"
    assert bool(lag_row["included"]) is True
    assert lag_row["lag"] == 1


def test_feature_matrix_can_force_current_factor_with_positive_lags() -> None:
    matrix = mf.feature_engineering.feature_matrix(
        _processed(),
        specification="F",
        columns=["x1", "x2"],
        lags=[1, 2],
        n_factors=1,
        fit_policy="full_sample",
        min_train_size=3,
        include_current_factor=True,
        warn_full_sample=False,
    )
    without_current = mf.feature_engineering.feature_matrix(
        _processed(),
        specification="F",
        columns=["x1", "x2"],
        lags=[1, 2],
        n_factors=1,
        fit_policy="full_sample",
        min_train_size=3,
        include_current_factor=False,
        warn_full_sample=False,
    )

    assert "F__F1_lag0" in matrix.columns
    assert "F__F1_lag0" not in without_current.columns
    stage = matrix.attrs["macroforecast_metadata"]["feature_engineering_feature_matrix"]
    assert stage["include_current_factor"] is True
    assert stage["factor_lags"] == [0, 1, 2]


def test_feature_matrix_requires_level_data_for_level_block() -> None:
    with pytest.raises(ValueError, match="level_data is required"):
        mf.feature_engineering.feature_matrix(_processed(), specification="X-LEVEL")


def test_build_features_accepts_composed_feature_steps_and_growth_target() -> None:
    features = mf.feature_engineering.build_features(
        _processed(),
        target_transform="growth",
        feature_steps=[
            mf.feature_engineering.moving_average_step(name="ma", columns=["x1"], windows=[1, 2], include=False),
            mf.feature_engineering.pca_step(name="ma_pc", input="ma", n_components=1, min_train_size=2),
            mf.feature_engineering.lag_step(name="ma_pc_lag", input="ma_pc", lags=[1]),
        ],
    )

    assert "ma_pc1" in features.X.columns
    assert "ma_pc1_lag1" in features.X.columns
    assert list(features.y.columns) == ["target_growth_h1", "target_growth_h2"]
    assert "feature_engineering" in features.metadata


def test_build_features_supports_direct_average_target() -> None:
    features = mf.feature_engineering.build_features(
        _processed(),
        target_transform="average_change",
        lags=(0,),
    )

    assert list(features.y.columns) == ["target_average_change_h1", "target_average_change_h2"]
    assert features.metadata["feature_engineering"]["target_mode"] == "direct"


def test_build_features_supports_path_target_mode() -> None:
    features = mf.feature_engineering.build_features(
        _processed(),
        target_transform="growth",
        target_mode="path",
        lags=(0,),
    )

    assert list(features.y.columns) == ["target_growth_step1", "target_growth_step2"]
    stage = features.metadata["feature_engineering"]
    assert stage["target_mode"] == "path"
    assert stage["path_target_columns_by_horizon"]["h2"]["target"] == list(features.y.columns)
    target_meta = features.target_metadata.set_index("target_column")
    assert target_meta.loc["target_growth_step1", "mode"] == "path"
    assert target_meta.loc["target_growth_step1", "used_for_horizons"] == "1,2"
    assert features.y.attrs["macroforecast_target_metadata"].equals(features.target_metadata)


def test_build_features_accepts_feature_matrix_specification() -> None:
    features = mf.feature_engineering.build_features(
        _processed(),
        feature_specification="F-X-MARX",
        lags=(0, 1),
        max_lag=2,
        n_factors=1,
        feature_fit_policy="full_sample",
        feature_min_train_size=3,
        feature_warn_full_sample=False,
    )

    assert "F__F1_lag0" in features.X.columns
    assert "X__x1_lag1" in features.X.columns
    assert "MARX__x1_ma2_lag1" in features.X.columns
    stage = features.metadata["feature_engineering"]
    assert stage["feature_specification"] == "F-X-MARX"
    assert stage["feature_matrix"]["fit_policy"] == "full_sample"
    marx_row = features.feature_metadata.loc[features.feature_metadata["feature"] == "MARX__x1_ma2_lag1"].iloc[0]
    assert marx_row["block"] == "MARX"
    assert marx_row["source"] == "x1"


def test_build_features_combines_path_targets_with_paper_feature_specification() -> None:
    features = mf.feature_engineering.build_features(
        _processed(),
        target_mode="path",
        target_transform="change",
        feature_specification="F-X-MAF",
        lags=(0, 1),
        max_lag=2,
        n_factors=1,
        n_maf_components=1,
        feature_fit_policy="full_sample",
        feature_min_train_size=3,
        feature_warn_full_sample=False,
    )

    assert "F__F1_lag0" in features.X.columns
    assert "MAF__x1_maf1" in features.X.columns
    assert list(features.y.columns) == ["target_change_step1", "target_change_step2"]
    assert features.metadata["feature_engineering"]["target_mode"] == "path"


def test_build_features_uses_spec_choices_and_preserves_metadata() -> None:
    features = mf.feature_engineering.build_features(
        _processed(),
        lags=(0, 1),
        rolling_windows=(2,),
        add_time=True,
        time_month=True,
    )

    assert isinstance(features, mf.feature_engineering.FeatureSet)
    assert features.X.index.equals(features.y.index)
    assert features.targets == ("target",)
    assert features.horizons == (1, 2)
    assert features.predictors == ("x1", "x2")
    assert "x1_lag0" in features.X.columns
    assert "x1_roll2_mean" in features.X.columns
    assert "month_01" in features.X.columns
    assert "feature_engineering" in features.metadata
    assert features.metadata["feature_engineering"]["output"]["n_features"] == features.X.shape[1]
    assert set(["feature", "operation", "source", "parameter", "block", "included"]).issubset(
        features.feature_metadata.columns
    )
    assert features.feature_metadata.attrs["macroforecast_metadata_schema"]["kind"] == "feature_metadata"
    assert not features.feature_metadata["included"].isna().any()
    assert set(features.feature_metadata.loc[features.feature_metadata["operation"] == "time", "source"]) == {"date"}
    assert set(["target_column", "source", "mode", "transform", "formula"]).issubset(
        features.target_metadata.columns
    )
    X, y, metadata = features
    assert X is features.X
    assert y is features.y
    assert metadata is features.metadata


def test_build_features_warns_without_preprocessing_metadata() -> None:
    with pytest.warns(UserWarning, match="PreprocessedData"):
        features = mf.feature_engineering.build_features(_panel(), target="target", predictors=["x1"], horizons=[1])

    assert list(features.X.columns) == ["x1_lag0", "x1_lag1"]


def test_build_features_rejects_target_as_predictor() -> None:
    with pytest.raises(ValueError, match="must not include target"):
        mf.feature_engineering.build_features(_processed(), predictors=["target", "x1"])


def test_build_features_rejects_single_predictor_string() -> None:
    with pytest.raises(TypeError, match="iterable of strings"):
        mf.feature_engineering.build_features(_processed(), predictors="x1")


def test_top_level_exports_features() -> None:
    assert mf.FeatureSet is mf.feature_engineering.FeatureSet
    assert mf.adaptive_ma_rf_features is mf.feature_engineering.adaptive_ma_rf_features
    assert mf.asymmetric_trim_features is mf.feature_engineering.asymmetric_trim_features
    assert mf.average_target is mf.feature_engineering.average_target
    assert mf.build_features is mf.feature_engineering.build_features
    assert mf.compose_features is mf.feature_engineering.compose_features
    assert mf.direct_target is mf.feature_engineering.direct_target
    assert mf.dfm_features is mf.feature_engineering.dfm_features
    assert mf.boruta_selection is mf.feature_engineering.boruta_selection
    assert mf.correlation_selection is mf.feature_engineering.correlation_selection
    assert mf.genetic_selection is mf.feature_engineering.genetic_selection
    assert mf.lasso_path_selection is mf.feature_engineering.lasso_path_selection
    assert mf.lasso_selection is mf.feature_engineering.lasso_selection
    assert mf.rfe_selection is mf.feature_engineering.rfe_selection
    assert mf.stability_selection is mf.feature_engineering.stability_selection
    assert mf.variance_selection is mf.feature_engineering.variance_selection
    assert mf.feature_matrix is mf.feature_engineering.feature_matrix
    assert mf.group_pca is mf.feature_engineering.group_pca
    assert mf.hamilton_filter_features is mf.feature_engineering.hamilton_filter_features
    assert mf.hamilton_step is mf.feature_engineering.hamilton_step
    assert mf.lag is mf.feature_engineering.lag
    assert mf.maf_features is mf.feature_engineering.maf_features
    assert mf.marx_step is mf.feature_engineering.marx_step
    assert mf.moving_average_ladder is mf.feature_engineering.moving_average_ladder
    assert mf.nystroem_step is mf.feature_engineering.nystroem_step
    assert mf.partial_least_squares_features is mf.feature_engineering.partial_least_squares_features
    assert mf.partial_least_squares_step is mf.feature_engineering.partial_least_squares_step
    assert mf.random_projection_step is mf.feature_engineering.random_projection_step
    assert mf.sliced_inverse_regression_features is mf.feature_engineering.sliced_inverse_regression_features
    assert mf.sliced_inverse_regression_step is mf.feature_engineering.sliced_inverse_regression_step
    assert mf.sparse_pca_chen_rohe_features is mf.feature_engineering.sparse_pca_chen_rohe_features
    assert mf.sparse_pca_chen_rohe_step is mf.feature_engineering.sparse_pca_chen_rohe_step
    assert mf.transform_step is mf.feature_engineering.transform_step
    assert mf.varimax_features is mf.feature_engineering.varimax_features
    assert mf.varimax_step is mf.feature_engineering.varimax_step
    assert mf.wavelet_features is mf.feature_engineering.wavelet_features
    assert mf.seasonal_lag_step is mf.feature_engineering.seasonal_lag_step
    assert mf.season_dummy_step is mf.feature_engineering.season_dummy_step
    assert mf.fourier_step is mf.feature_engineering.fourier_step
    assert mf.time_step is mf.feature_engineering.time_step
    assert mf.polynomial_step is mf.feature_engineering.polynomial_step
    assert mf.interaction_step is mf.feature_engineering.interaction_step
    assert mf.path_targets is mf.feature_engineering.path_targets
    assert mf.pca_then_lags is mf.feature_engineering.pca_then_lags
    assert not hasattr(mf.feature_engineering, "make_target")
    assert not hasattr(mf.feature_engineering, "make_feature_matrix")
    assert not hasattr(mf.feature_engineering, "feature_selection_features")
    assert not hasattr(mf.feature_engineering, "feature_selection_step")
    assert not hasattr(mf.feature_engineering, "group_pca_features")
    assert not hasattr(mf, "make_target")
    assert not hasattr(mf, "feature_selection_features")
    assert not hasattr(mf, "feature_selection_step")
    assert not hasattr(mf, "group_pca_features")


def test_feature_engineering_runner_safe_coverage_is_explicit() -> None:
    runner_safe_steps = {
        "lag_step",
        "rolling_step",
        "moving_average_step",
        "marx_step",
        "transform_step",
        "seasonal_lag_step",
        "season_dummy_step",
        "fourier_step",
        "time_step",
        "polynomial_step",
        "interaction_step",
        "scale_step",
        "pca_step",
        "sparse_pca_chen_rohe_step",
        "varimax_step",
        "group_pca_step",
        "maf_step",
        "hamilton_step",
        "random_projection_step",
        "nystroem_step",
        "partial_least_squares_step",
        "sliced_inverse_regression_step",
    }
    runner_safe_step_methods = {
        "variance_selection",
        "correlation_selection",
        "lasso_selection",
        "lasso_path_selection",
        "rfe_selection",
        "boruta_selection",
        "stability_selection",
        "genetic_selection",
    }
    direct_only = {
        "mixed_frequency_lags",
        "hp_filter_features",
        "savitzky_golay_features",
    }

    for name in runner_safe_steps:
        assert callable(getattr(mf.feature_engineering, name))
    for method in runner_safe_step_methods:
        assert isinstance(
            mf.feature_engineering.feature_spec(
                target="target",
                horizon=1,
                predictors=["x1"],
                steps=[{"name": method, "method": method, "n_features": 1}],
            ),
            mf.feature_engineering.FeatureSpec,
        )
    for name in direct_only:
        assert callable(getattr(mf.feature_engineering, name))

    unsupported_step_methods = [
        "mixed_frequency_lags",
        "hp_filter",
        "savitzky_golay",
    ]
    for method in unsupported_step_methods:
        with pytest.raises(ValueError, match="feature method"):
            mf.feature_engineering.feature_spec(
                target="target",
                horizon=1,
                predictors=["x1"],
                steps=[{"name": method, "method": method}],
            )
