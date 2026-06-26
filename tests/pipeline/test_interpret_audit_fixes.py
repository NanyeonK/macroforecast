"""Regression tests for the independent audit of interpret_pipeline."""
import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import (
    Arm, EvalSpec, InterpretSpec, interpret_pipeline, pipeline_spec, run_pipeline,
)

# Every test here drives SHAP interpretation end to end. Without the optional
# 'interpretation' extra (shap), interpret_pipeline returns a graceful error
# table instead of importances, so skip the module — matching the [ci] install
# and the importorskip convention used by the other optional-dependency tests.
pytest.importorskip("shap")


def _bundle(n=120):
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    rng = np.random.default_rng(0)
    x1 = rng.standard_normal(n).cumsum()
    x2 = rng.standard_normal(n).cumsum()
    y1 = 0.9 * np.r_[0, x1[:-1]] + rng.standard_normal(n) * 0.1   # y1 driven by x1
    y2 = 0.9 * np.r_[0, x2[:-1]] + rng.standard_normal(n) * 0.1   # y2 driven by x2
    frame = pd.DataFrame({"y1": y1, "y2": y2, "x1": x1, "x2": x2}, index=idx)
    return mf.data.custom_dataset(frame, transform_codes={c: 1 for c in ("y1", "y2", "x1", "x2")})


def _spec(targets, interpret, estimation=None, **over):
    feats = mf.feature_engineering.feature_spec(target="y1", predictors=["x1", "x2"], lags=(0, 1), target_lags=(0, 1))
    est = estimation or mf.window.estimation_expanding(min_size=48)
    w = mf.window.spec(estimation=est, val=mf.window.val_last_block(size=12), test=mf.window.test_origins(horizon=1, step=6))
    kw = dict(
        data=_bundle(), targets=targets, horizons=[1], window=w,
        arms=[Arm("AR", model="ar"), Arm("RF", model="random_forest", features=feats, interpret=interpret)],
        evaluation=EvalSpec(benchmark="AR"), save_models=False,
    )
    kw.update(over)
    return pipeline_spec(**kw)


def test_which_fit_validation_rejects_unknown():
    report = run_pipeline(_spec([mf.pipeline.TargetSpec("y1", transform="level")], InterpretSpec(methods=("shap",))))
    with pytest.raises(ValueError):
        interpret_pipeline(report, which_fit="totally_invalid")


def test_origin_mean_differs_from_final():
    report = run_pipeline(_spec([mf.pipeline.TargetSpec("y1", transform="level")], InterpretSpec(methods=("shap",))))
    final = interpret_pipeline(report, methods=("shap",), which_fit="final")
    omean = interpret_pipeline(report, methods=("shap",), which_fit="origin_mean")
    f = next(iter(final["RF"].values()))["shap"].set_index("feature")["importance"] if "feature" in next(iter(final["RF"].values()))["shap"].columns else None
    # at minimum, origin_mean must run and return a per-feature table (not identical object/ignored)
    o_tab = next(iter(omean["RF"].values()))["shap"]
    f_tab = next(iter(final["RF"].values()))["shap"]
    assert not o_tab.empty
    # they should not be byte-identical (origin_mean averages multiple refits)
    assert not o_tab.equals(f_tab)


def test_multitarget_interpretation_retargets():
    spec = _spec(
        [mf.pipeline.TargetSpec("y1", transform="level"), mf.pipeline.TargetSpec("y2", transform="level")],
        InterpretSpec(methods=("shap",)),
    )
    report = run_pipeline(spec)
    out = interpret_pipeline(report)
    rf = out["RF"]
    # two target-keyed entries, and they must differ (y1 driven by x1, y2 by x2)
    keys = list(rf.keys())
    assert len(keys) == 2
    t1 = rf[keys[0]]["shap"]; t2 = rf[keys[1]]["shap"]
    assert not t1.equals(t2)


def test_one_failing_method_does_not_crash_all():
    # GOOD arm with shap, BAD arm requesting an unsupported method -> error frame, not abort
    feats = mf.feature_engineering.feature_spec(target="y1", predictors=["x1", "x2"], lags=(0, 1), target_lags=(0, 1))
    spec = pipeline_spec(
        data=_bundle(n=110), targets=[mf.pipeline.TargetSpec("y1", transform="level")], horizons=[1],
        window=mf.window.spec(estimation=mf.window.estimation_expanding(min_size=48),
                              val=mf.window.val_last_block(size=12), test=mf.window.test_origins(horizon=1, step=6)),
        arms=[Arm("AR", model="ar"),
              Arm("GOOD", model="random_forest", features=feats, interpret=InterpretSpec(methods=("shap",))),
              Arm("BAD", model="random_forest", features=feats, interpret=InterpretSpec(methods=("nonexistent_method",)))],
        evaluation=EvalSpec(benchmark="AR"), save_models=False,
    )
    report = run_pipeline(spec)
    out = interpret_pipeline(report)  # must not raise
    assert "GOOD" in out
    good = next(iter(out["GOOD"].values()))["shap"]
    assert not good.empty and "error" not in good.columns
    # BAD arm degrades to an error frame, does not abort the run
    bad = next(iter(out["BAD"].values()))["nonexistent_method"]
    assert "error" in bad.columns
