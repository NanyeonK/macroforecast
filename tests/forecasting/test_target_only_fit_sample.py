"""A target-only model's fit sample must not be truncated by predictor NaNs.

Every model whose ``ModelSpec.input_kind`` is ``"target"`` takes exactly one
positional argument -- the target. ``hist_mean(y)``, ``naive(y)``, ``arima(y)``:
there is no parameter through which X could reach them. Yet the fit sample was
built by ``pd.concat([X, y], axis=1).dropna()``, so a predictor that is NaN over
the first rows of the window removed those rows from **y** as well.

The benchmark is where this bites hardest. `hist_mean` is the canonical
prevailing-mean benchmark, and its value is the mean of the target over the fit
window; drop the first rows of that window and every ``R2_OS`` in the run moves,
with the size of the move set by whichever contender happens to carry the longest
lag. The same benchmark then scores differently depending on which arms share the
bundle, which is the one thing a benchmark must not do.

Supervised models are untouched: their X is part of the fit, so a row with a NaN
predictor is genuinely unusable for them.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline

N = 160
FIRST_ORIGIN = 120


def _panel(leading_nan: int) -> pd.DataFrame:
    """Target complete throughout; one predictor missing its first rows."""
    idx = pd.date_range("1990-01-31", periods=N, freq="ME", name="date")
    rng = np.random.default_rng(3)
    y = pd.Series(np.cumsum(rng.normal(size=N)) * 0.1 + 2.0, index=idx, name="y")
    x = pd.Series(rng.normal(size=N), index=idx, name="x")
    if leading_nan:
        x.iloc[:leading_nan] = np.nan
    return pd.DataFrame({"y": y, "x": x}, index=idx)


def _first_benchmark_forecast(leading_nan: int) -> tuple[float, pd.DataFrame]:
    frame = _panel(leading_nan)
    bundle = mf.data.custom_dataset(frame, transform_codes={c: 1 for c in frame.columns})
    window = mf.window.from_cutoffs(
        test_start=frame.index[FIRST_ORIGIN], horizon=1, embargo=0,
        val_method="expanding", val_min_train_size=12,
    )
    spec = pipeline_spec(
        data=bundle,
        targets=[TargetSpec("y", transform="level")],
        horizons=[1],
        window=window,
        # NOTE: predictors deliberately left unset, which resolves to the whole
        # panel. That is the documented default and the shape a caller writes by
        # accident; the benchmark must be insensitive to it.
        arms=[
            Arm("HA", model="hist_mean",
                features=mf.feature_engineering.feature_spec(target="y", target_lags=(1,)),
                is_benchmark=True),
            Arm("OLS", model="ols",
                features=mf.feature_engineering.feature_spec(
                    target="y", predictors=["x"], lags=0, target_lags=None)),
        ],
        evaluation=EvalSpec(benchmark="HA", metrics=("r2_oos",), tests=()),
        save_models=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = run_pipeline(spec)
    fc = report.forecasts
    ha = fc[fc["contender"] == "HA"].sort_values("date")
    return float(ha["prediction"].iloc[0]), fc


def test_benchmark_is_unaffected_by_a_predictor_gap() -> None:
    """The same target, the same window: the benchmark must not move."""
    clean, _ = _first_benchmark_forecast(0)
    gapped, _ = _first_benchmark_forecast(9)
    assert clean == pytest.approx(gapped, rel=0, abs=1e-12), (
        f"benchmark moved {gapped - clean:+.6g} because a PREDICTOR it never "
        f"reads was missing its first 9 rows"
    )


def test_benchmark_equals_the_prevailing_mean_of_the_fit_window() -> None:
    """Not just stable -- correct: the mean of the h=1 target over the fit window.

    The averaged quantity is the TARGET COLUMN, which for a direct h = 1 run is
    ``y`` shifted one step: the observation at position 0 is never anyone's
    target, because no origin precedes it. So the benchmark at an origin that
    forecasts position ``pos`` is the mean of ``y[1 : pos]``, not ``y[0 : pos]``.
    The distinction is one observation out of a hundred-plus here, but pinning
    the wrong one would make this test pass only by accident.
    """

    frame = _panel(9)
    gapped, fc = _first_benchmark_forecast(9)
    ha = fc[fc["contender"] == "HA"].sort_values("date")
    pos = frame.index.get_loc(ha["date"].iloc[0])
    expected = float(frame["y"].iloc[1:pos].mean())
    assert gapped == pytest.approx(expected, rel=1e-12), (
        "the prevailing mean must average every target observation in the fit "
        "window, including the rows a predictor gap used to remove"
    )
    # and it must NOT be the truncated mean the predictor gap used to produce
    truncated = float(frame["y"].iloc[10:pos].mean())
    assert abs(gapped - truncated) > 1e-6, "still averaging the truncated window"


def test_supervised_arms_still_drop_rows_with_missing_predictors() -> None:
    """The complement: X IS part of a supervised fit, so its NaNs still count."""
    from macroforecast.forecasting.runner import _slice_feature_set
    from macroforecast.feature_engineering import FeatureSet

    idx = pd.date_range("1990-01-31", periods=6, freq="ME")
    X = pd.DataFrame({"a": [np.nan, np.nan, 1.0, 2.0, 3.0, 4.0]}, index=idx)
    y = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=idx)
    fs = FeatureSet(
        X=X, y=y, metadata={}, feature_metadata=pd.DataFrame(),
        target_metadata=pd.DataFrame(), target="y", targets=("y",),
        horizons=(1,), predictors=("a",),
    )
    supervised = _slice_feature_set(fs, idx, drop_missing=True, target_only=False)
    assert len(supervised.y) == 4, "supervised fit must still drop the NaN rows"

    target_only = _slice_feature_set(fs, idx, drop_missing=True, target_only=True)
    assert len(target_only.y) == 6, "a target-only fit keeps every target row"
    assert len(target_only.X) == len(target_only.y), "X and y stay aligned"


def test_a_nan_target_row_is_dropped_either_way() -> None:
    """Missing TARGET values are unusable for any model, target-only included."""
    from macroforecast.forecasting.runner import _slice_feature_set
    from macroforecast.feature_engineering import FeatureSet

    idx = pd.date_range("1990-01-31", periods=5, freq="ME")
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
    y = pd.DataFrame({"y": [np.nan, 2.0, 3.0, np.nan, 5.0]}, index=idx)
    fs = FeatureSet(
        X=X, y=y, metadata={}, feature_metadata=pd.DataFrame(),
        target_metadata=pd.DataFrame(), target="y", targets=("y",),
        horizons=(1,), predictors=("a",),
    )
    out = _slice_feature_set(fs, idx, drop_missing=True, target_only=True)
    assert len(out.y) == 3
    assert not out.y.isna().to_numpy().any()


def _target_only_policy_predictions(
    leading_nan: int,
    policy: str,
) -> pd.Series:
    frame = _panel(leading_nan)
    window = mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=24),
        val=mf.window.val_last_block(size=8),
        test=mf.window.test_origins(
            first_origin=frame.index[FIRST_ORIGIN],
            last_origin=frame.index[FIRST_ORIGIN + 2],
            horizon=1,
        ),
    )
    result = mf.forecasting.run(
        frame,
        "hist_mean",
        window=window,
        target="y",
        horizon=1,
        forecast_policy=policy,
        target_transform=(
            "average_value" if policy == "direct_average" else "value"
        ),
        model_selection={"hist_mean": None},
        save_models=False,
    ).to_frame()
    return result.set_index("origin")["prediction"].sort_index()


def test_path_policy_preserves_target_only_sample_with_predictor_gaps() -> None:
    """At h=1, direct and path use the same target-only estimation sample."""
    frame = _panel(12)
    direct = _target_only_policy_predictions(12, "direct_average")
    path = _target_only_policy_predictions(12, "path_average")
    assert not direct.empty
    pd.testing.assert_series_equal(
        path,
        direct,
        check_names=False,
        rtol=0,
        atol=1e-12,
    )
    expected = float(frame["y"].iloc[1 : FIRST_ORIGIN + 1].mean())
    assert path.iloc[0] == pytest.approx(expected, rel=0, abs=1e-12)
