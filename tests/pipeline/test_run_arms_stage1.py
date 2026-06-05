"""Stage 1: run_arms executes arms into the master forecast frame."""
import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, pipeline_spec, run_arms


def _bundle(n=72):
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, n)
    frame = pd.DataFrame(
        {"y": 1.0 + 2.0 * x + rng.standard_normal(n) * 0.05, "x1": x},
        index=idx,
    )
    # y is tcode 1 (level / stationary here) to keep the target simple
    return mf.data.custom_dataset(frame, transform_codes={"y": 1, "x1": 1})


def _window():
    return mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=36),
        val=mf.window.val_last_block(size=12),
        test=mf.window.test_origins(horizon=1, step=6),
    )


def _spec(**over):
    feats = mf.feature_engineering.feature_spec(target="y", predictors=["x1"], lags=1, target_lags=(0, 1))
    kw = dict(
        data=_bundle(), targets=["y"], horizons=[1, 3], window=_window(),
        arms=[Arm("AR", model="ar", features=feats), Arm("OLS", model="ols", features=feats)],
        evaluation=EvalSpec(benchmark="AR"), save_models=False,
    )
    kw.update(over)
    return pipeline_spec(**kw)


def test_master_frame_has_required_columns_and_tags():
    master = run_arms(_spec())
    assert not master.empty
    for col in ("arm", "contender", "model", "target", "horizon", "prediction", "actual"):
        assert col in master.columns
    # both arms present as contenders (single model -> contender == arm name)
    assert set(master["arm"]) == {"AR", "OLS"}
    assert set(master["contender"]) == {"AR", "OLS"}
    assert set(master["horizon"]) == {1, 3}


def test_multi_model_arm_yields_arm_colon_model_contenders():
    feats = mf.feature_engineering.feature_spec(target="y", predictors=["x1"], lags=1, target_lags=(0, 1))
    spec = _spec(
        arms=[Arm("bench", model="ar", features=feats), Arm("ML", model=["ols", "ridge"], features=feats)],
        evaluation=EvalSpec(benchmark="bench"),
    )
    master = run_arms(spec)
    ml = set(master.loc[master["arm"] == "ML", "contender"])
    assert ml == {"ML:ols", "ML:ridge"}


def test_predictions_are_finite_on_each_contender():
    master = run_arms(_spec())
    assert master["prediction"].notna().any()
    assert np.isfinite(master["prediction"].dropna().to_numpy()).all()
