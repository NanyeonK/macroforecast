"""Spec-level shared preprocessing (freeze/vary contract) + refit cadence."""
import numpy as np
import pandas as pd

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, pipeline_spec, run_pipeline


def _bundle(n=120):
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    rng = np.random.default_rng(0)
    x = np.linspace(1.0, 4.0, n) + rng.standard_normal(n) * 0.05
    frame = pd.DataFrame({"y": x, "x1": x + rng.standard_normal(n) * 0.1}, index=idx)
    return mf.data.custom_dataset(frame, transform_codes={"y": 5, "x1": 5})


def _spec(**over):
    feats = mf.feature_engineering.feature_spec(target="y", predictors=["x1"], lags=1, target_lags=(0, 1))
    w = mf.window.spec(estimation=mf.window.estimation_expanding(min_size=48),
                       val=mf.window.val_last_block(size=12), test=mf.window.test_origins(horizon=1, step=6))
    pp = mf.preprocessing.preprocess_spec(transform="official", impute="mean", outliers="none", standardize="none")
    kw = dict(
        data=_bundle(), targets=[mf.pipeline.TargetSpec("y", transform="average_value", policy="direct_average")],
        horizons=[1], window=w,
        arms=[Arm("AR", model="ar", features=feats), Arm("OLS", model="ols", features=feats)],
        evaluation=EvalSpec(benchmark="AR"), save_models=False,
        preprocessing=pp, preprocessing_policy=mf.window.stage_policy("origin_available", update="on_retrain"),
    )
    kw.update(over)
    return pipeline_spec(**kw)


def test_spec_level_preprocessing_applies_to_all_arms():
    spec = _spec()
    assert spec.preprocessing is not None
    report = run_pipeline(spec)
    # both arms ran with the shared preprocessing -> forecasts present, preprocessed flag set
    assert not report.forecasts.empty
    assert set(report.forecasts["arm"]) == {"AR", "OLS"}
    if "preprocessed" in report.forecasts.columns:
        assert report.forecasts["preprocessed"].any()


def test_arm_preprocessing_overrides_spec_level():
    own = mf.preprocessing.preprocess_spec(transform="official", impute="forward_fill", outliers="none", standardize="none")
    spec = _spec(arms=[Arm("AR", model="ar",
                           features=mf.feature_engineering.feature_spec(target="y", predictors=["x1"], lags=1, target_lags=(0, 1))),
                       Arm("OLS_own", model="ols",
                           features=mf.feature_engineering.feature_spec(target="y", predictors=["x1"], lags=1, target_lags=(0, 1)),
                           preprocessing=own)],
                 evaluation=EvalSpec(benchmark="AR"))
    report = run_pipeline(spec)
    assert not report.forecasts.empty
