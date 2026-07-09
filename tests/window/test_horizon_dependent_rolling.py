from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def _medeiros_size(base: int, horizon: int) -> int:
    return base - horizon - 4 - 1


def _deterministic_panel() -> pd.DataFrame:
    idx = pd.date_range("2000-01-31", periods=96, freq="ME", name="date")
    x = np.linspace(0.0, 1.0, 96)
    return pd.DataFrame(
        {
            "y": 1.0 + 2.0 * x + 0.1 * np.sin(np.arange(96)),
            "x1": x,
        },
        index=idx,
    )


def _ols_features() -> mf.feature_engineering.FeatureSpec:
    return mf.feature_engineering.feature_spec(
        target="y",
        horizon=3,
        predictors=["x1"],
        lags=(0,),
    )


def _comparable_forecasts(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "origin_pos",
        "date",
        "horizon",
        "target",
        "model",
        "actual",
        "prediction",
    ]
    return (
        frame.loc[:, columns]
        .sort_values(["origin_pos", "date", "horizon", "target", "model"])
        .reset_index(drop=True)
    )


@pytest.mark.parametrize(
    ("horizon", "expected_r", "expected_start"),
    [(1, 34, 26), (3, 32, 28), (6, 29, 31)],
)
def test_horizon_size_rule_resolves_rolling_size_per_horizon(
    horizon: int,
    expected_r: int,
    expected_start: int,
) -> None:
    window = mf.window.from_cutoffs(
        test_start=60,
        test_end=60,
        mode="rolling",
        estimation_size=40,
        estimation_size_rule=_medeiros_size,
        horizon=horizon,
    )

    plan = window.origins(pd.RangeIndex(100))
    row = plan.iloc[0]

    assert int(row["horizon"]) == horizon
    assert int(row["estimation_end_pos"]) == 59
    assert int(row["estimation_start_pos"]) == expected_start
    assert int(row["n_estimation"]) == expected_r
    assert window.estimation.to_dict()["size_rule"] == "_medeiros_size"


@pytest.mark.parametrize(
    ("horizon", "expected_r", "expected_start"),
    [(1, 34, 26), (3, 32, 28), (6, 29, 31)],
)
def test_size_by_horizon_resolves_rolling_size_per_horizon(
    horizon: int,
    expected_r: int,
    expected_start: int,
) -> None:
    window = mf.window.from_cutoffs(
        test_start=60,
        test_end=60,
        mode="rolling",
        estimation_size_by_horizon={1: 34, 3: 32, 6: 29},
        horizon=horizon,
    )

    plan = window.origins(pd.RangeIndex(100))
    row = plan.iloc[0]

    assert int(row["horizon"]) == horizon
    assert int(row["estimation_end_pos"]) == 59
    assert int(row["estimation_start_pos"]) == expected_start
    assert int(row["n_estimation"]) == expected_r
    assert window.estimation.to_dict()["size_by_horizon"] == {1: 34, 3: 32, 6: 29}


def test_map_only_rolling_window_uses_horizon_size_for_default_first_origin() -> None:
    window = mf.window.spec(
        estimation=mf.window.estimation_rolling(size_by_horizon={3: 32}),
        val=mf.window.val_last_block(size=8),
        test=mf.window.test_origins(horizon=3),
    )

    first = window.origins(pd.RangeIndex(80)).iloc[0]

    assert int(first["origin_pos"]) == 32
    assert int(first["estimation_start_pos"]) == 0
    assert int(first["estimation_end_pos"]) == 31
    assert int(first["n_estimation"]) == 32


def test_fixed_rolling_origin_plan_and_metadata_are_unchanged() -> None:
    window = mf.window.from_cutoffs(
        test_start=40,
        test_end=42,
        mode="rolling",
        estimation_size=24,
        horizon=3,
        step=1,
    )
    columns = [
        "origin_pos",
        "estimation_start_pos",
        "estimation_end_pos",
        "n_estimation",
        "test_start_pos",
        "test_end_pos",
    ]
    expected = pd.DataFrame(
        {
            "origin_pos": [40, 41, 42],
            "estimation_start_pos": [16, 17, 18],
            "estimation_end_pos": [39, 40, 41],
            "n_estimation": [24, 24, 24],
            "test_start_pos": [40, 41, 42],
            "test_end_pos": [42, 43, 44],
        },
    )

    plan = window.origins(pd.RangeIndex(80))

    pd.testing.assert_frame_equal(
        plan.loc[:, columns].reset_index(drop=True),
        expected,
    )
    assert window.estimation.to_dict() == {
        "mode": "rolling",
        "start": None,
        "end": None,
        "min_size": None,
        "size": 24,
        "embargo": 0,
        "retrain_every": 1,
    }


def test_horizon_rule_matching_fixed_r_reproduces_same_ols_forecasts() -> None:
    panel = _deterministic_panel()
    common = {
        "test_start": panel.index[60],
        "test_end": panel.index[64],
        "mode": "rolling",
        "val_method": "last_block",
        "val_size": 8,
    }
    rule_window = mf.window.from_cutoffs(
        **common,
        estimation_size=40,
        estimation_size_rule=_medeiros_size,
    )
    fixed_window = mf.window.from_cutoffs(**common, estimation_size=32)

    rule = mf.forecasting.run(
        panel,
        "ols",
        window=rule_window,
        features=_ols_features(),
        horizon=3,
        save_models=False,
    ).to_frame()
    fixed = mf.forecasting.run(
        panel,
        "ols",
        window=fixed_window,
        features=_ols_features(),
        horizon=3,
        save_models=False,
    ).to_frame()

    rule_cmp = _comparable_forecasts(rule)
    fixed_cmp = _comparable_forecasts(fixed)

    pd.testing.assert_frame_equal(
        rule_cmp.drop(columns=["prediction"]),
        fixed_cmp.drop(columns=["prediction"]),
    )
    np.testing.assert_allclose(
        rule_cmp["prediction"].to_numpy(),
        fixed_cmp["prediction"].to_numpy(),
        rtol=0,
        atol=1e-12,
    )


def test_fixed_rolling_forecasts_keep_golden_predictions() -> None:
    panel = _deterministic_panel()
    window = mf.window.from_cutoffs(
        test_start=panel.index[60],
        test_end=panel.index[64],
        mode="rolling",
        estimation_size=32,
        val_method="last_block",
        val_size=8,
    )

    forecasts = mf.forecasting.run(
        panel,
        "ols",
        window=window,
        features=_ols_features(),
        horizon=3,
        save_models=False,
    ).to_frame()
    comparable = _comparable_forecasts(forecasts)

    assert comparable["origin_pos"].tolist() == [60, 61, 62, 63, 64]
    np.testing.assert_allclose(
        comparable["actual"].to_numpy(),
        np.array([
            2.3430513595039644,
            2.4393710248723104,
            2.451103920580589,
            2.3868185688081294,
            2.3249743178919413,
        ]),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        comparable["prediction"].to_numpy(),
        np.array([
            2.3318641714605186,
            2.3342044499581185,
            2.3486476227712507,
            2.3812703958081247,
            2.4214352343707644,
        ]),
        rtol=0,
        atol=1e-12,
    )


def test_horizon_dependent_rolling_size_validation_errors() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        mf.window.estimation_rolling(
            size=40,
            size_rule=_medeiros_size,
            size_by_horizon={3: 32},
        )
    with pytest.raises(ValueError, match="estimation_size.*size_rule"):
        mf.window.from_cutoffs(
            test_start=10,
            mode="rolling",
            estimation_size_rule=_medeiros_size,
        )
    with pytest.raises(ValueError, match="positive integer"):
        mf.window.estimation_rolling(size_by_horizon={0: 32})
    with pytest.raises(ValueError, match="positive integer"):
        mf.window.estimation_rolling(size_by_horizon={3: 0})
    with pytest.raises(ValueError, match="only valid for rolling"):
        mf.window.EstimationWindow(mode="expanding", size_rule=_medeiros_size)
    with pytest.raises(ValueError, match="only valid for rolling"):
        mf.window.EstimationWindow(mode="fixed", size_by_horizon={3: 32})
    with pytest.raises(ValueError, match="mode='rolling'"):
        mf.window.from_cutoffs(
            test_start=10,
            mode="expanding",
            estimation_size_by_horizon={3: 32},
        )

    missing = mf.window.from_cutoffs(
        test_start=10,
        mode="rolling",
        estimation_size_by_horizon={1: 8},
        horizon=3,
    )
    with pytest.raises(ValueError, match="missing horizon 3"):
        missing.origins(pd.RangeIndex(40))

    non_positive = mf.window.from_cutoffs(
        test_start=10,
        mode="rolling",
        estimation_size=8,
        estimation_size_rule=lambda base, horizon: 0,
        horizon=3,
    )
    with pytest.raises(ValueError, match="horizon 3"):
        non_positive.origins(pd.RangeIndex(40))

    non_integer = mf.window.from_cutoffs(
        test_start=10,
        mode="rolling",
        estimation_size=8,
        estimation_size_rule=lambda base, horizon: 3.5,
        horizon=3,
    )
    with pytest.raises(ValueError, match="horizon 3"):
        non_integer.origins(pd.RangeIndex(40))
