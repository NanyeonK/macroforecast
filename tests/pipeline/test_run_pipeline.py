"""End-to-end: run_pipeline assembles a PipelineReport."""
import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import Arm, CombinationContender, EvalSpec, pipeline_spec, run_pipeline


def _bundle(n=96):
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, n)
    frame = pd.DataFrame({"y": 1.0 + 2.0 * x + rng.standard_normal(n) * 0.05, "x1": x}, index=idx)
    return mf.data.custom_dataset(frame, transform_codes={"y": 1, "x1": 1})


def _spec(**over):
    feats = mf.feature_engineering.feature_spec(target="y", predictors=["x1"], lags=1, target_lags=(0, 1))
    w = mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=36),
        val=mf.window.val_last_block(size=12),
        test=mf.window.test_origins(horizon=1, step=3),
    )
    kw = dict(
        data=_bundle(), targets=["y"], horizons=[1], window=w,
        arms=[Arm("AR", model="ar", features=feats), Arm("OLS", model="ols", features=feats)],
        evaluation=EvalSpec(benchmark="AR"), save_models=False,
        combinations=[CombinationContender(name="POOL", method="mean")],
    )
    kw.update(over)
    return pipeline_spec(**kw)


def test_run_pipeline_report_populated():
    report = run_pipeline(_spec())
    assert not report.forecasts.empty
    assert not report.accuracy.empty
    # combination contender present in the evaluated outputs
    assert "POOL" in set(report.accuracy["contender"])
    # provenance carries reproducibility info
    assert report.provenance["benchmark"] == "AR"
    assert report.provenance["seed"] == 42
    assert {"name", "policy", "transform", "tcode"} <= set(report.provenance["targets"][0])
    assert report.provenance["targets"][0]["tcode"] == 1
    # report is a single object tying everything together
    assert report.to_frame() is report.forecasts


def test_provenance_records_arms_and_horizons():
    report = run_pipeline(_spec())
    assert set(report.provenance["arms"]) == {"AR", "OLS"}
    assert report.provenance["horizons"] == [1]
    assert "POOL" in report.provenance["combinations"]


def test_leakage_audit_surfaces_embargo_warning():
    idx = pd.date_range("2000-01-31", periods=96, freq="ME", name="date")
    x = np.linspace(0.0, 1.0, 96)
    frame = pd.DataFrame({"y": 1.0 + 2.0 * x + np.random.default_rng(0).standard_normal(96) * 0.05, "x1": x}, index=idx)
    bundle = mf.data.custom_dataset(frame, transform_codes={"y": 1, "x1": 1})
    feats = mf.feature_engineering.feature_spec(target="y", predictors=["x1"], lags=1, target_lags=(0, 1))
    # multi-horizon window with default embargo -> pseudo-OOS warning expected
    w = mf.window.from_cutoffs(test_start=idx[60], horizon=3, val_method="expanding", val_min_train_size=24)
    spec = pipeline_spec(
        data=bundle, targets=[mf.pipeline.TargetSpec("y", transform="level")], horizons=[3], window=w,
        arms=[Arm("AR", model="ar", features=feats), Arm("OLS", model="ols", features=feats)],
        evaluation=EvalSpec(benchmark="AR"), save_models=False,
    )
    report = run_pipeline(spec)
    warns = report.leakage_audit.get("window_warnings") or []
    assert any("pseudo-out-of-sample" in w for w in warns)
