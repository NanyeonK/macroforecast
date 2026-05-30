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
    assert mf.average_target is mf.feature_engineering.average_target
    assert mf.build_features is mf.feature_engineering.build_features
    assert mf.compose_features is mf.feature_engineering.compose_features
    assert mf.direct_target is mf.feature_engineering.direct_target
    assert mf.feature_matrix is mf.feature_engineering.feature_matrix
    assert mf.group_pca is mf.feature_engineering.group_pca
    assert mf.lag is mf.feature_engineering.lag
    assert mf.maf_features is mf.feature_engineering.maf_features
    assert mf.moving_average_ladder is mf.feature_engineering.moving_average_ladder
    assert mf.path_targets is mf.feature_engineering.path_targets
    assert mf.pca_then_lags is mf.feature_engineering.pca_then_lags
    assert not hasattr(mf.feature_engineering, "make_target")
    assert not hasattr(mf.feature_engineering, "make_feature_matrix")
    assert not hasattr(mf.feature_engineering, "group_pca_features")
    assert not hasattr(mf, "make_target")
    assert not hasattr(mf, "group_pca_features")
