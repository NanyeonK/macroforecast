"""Regression tests for independent-audit findings on the pipeline."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import (
    Arm, CombinationContender, EvalSpec, apply_combinations, pipeline_spec, run_pipeline,
)
from macroforecast.pipeline.evaluate import accuracy_table
from macroforecast.pipeline.spec import PipelineSpec, ResolvedTarget


def _master_ragged():
    """Benchmark covers 40 origins; A has identical errors but drops the 20 hard ones."""
    rng = np.random.default_rng(0)
    n = 40
    origins = np.arange(n)
    actual = np.zeros(n)
    err = np.concatenate([rng.standard_normal(20) * 1.0, rng.standard_normal(20) * 5.0])
    rows = []
    for i in origins:
        rows.append({"target": "y", "horizon": 1, "origin": int(i), "date": i,
                     "contender": "BENCH", "prediction": actual[i] + err[i], "actual": actual[i]})
        if i < 20:  # A only reports the easy origins, with identical errors
            rows.append({"target": "y", "horizon": 1, "origin": int(i), "date": i,
                         "contender": "A", "prediction": actual[i] + err[i], "actual": actual[i]})
    return pd.DataFrame(rows)


def _spec_stub(benchmark="BENCH"):
    return PipelineSpec(
        data=None, targets=(ResolvedTarget("y", "direct", "level", 1, False),),
        horizons=(1,), window=None, arms=(Arm("BENCH", "ar"), Arm("A", "ols")),
        evaluation=EvalSpec(benchmark=benchmark),
    )


def test_accuracy_uses_common_sample_not_survivorship():
    acc = accuracy_table(_master_ragged(), _spec_stub())
    a_row = acc[acc["contender"] == "A"].iloc[0]
    # A is identical to the benchmark on every shared origin -> relative_mse must be ~1, not <1
    assert np.isclose(a_row["relative_mse"], 1.0, atol=1e-9)
    assert np.isclose(a_row["r2_oos"], 0.0, atol=1e-9)


def _bundle(n=96):
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    x = np.linspace(0.0, 1.0, n)
    frame = pd.DataFrame({"y": 1.0 + 2.0 * x + np.random.default_rng(0).standard_normal(n) * 0.05, "x1": x}, index=idx)
    return mf.data.custom_dataset(frame, transform_codes={"y": 1, "x1": 1})


def test_leakage_audit_fires_for_multihorizon_base_window():
    bundle = _bundle()
    idx = bundle.panel.index
    feats = mf.feature_engineering.feature_spec(target="y", predictors=["x1"], lags=1, target_lags=(0, 1))
    # base window has test.horizon = 1; the run uses horizons [1, 3] with default embargo=0
    w = mf.window.from_cutoffs(test_start=idx[60], horizon=1, embargo=0, val_method="expanding", val_min_train_size=24)
    spec = pipeline_spec(
        data=bundle, targets=[mf.pipeline.TargetSpec("y", transform="level")], horizons=[1, 3], window=w,
        arms=[Arm("AR", model="ar", features=feats), Arm("OLS", model="ols", features=feats)],
        evaluation=EvalSpec(benchmark="AR"), save_models=False,
    )
    report = run_pipeline(spec)
    warns = report.leakage_audit.get("window_warnings") or []
    # the h=3 sub-window with embargo 0 must surface the pseudo-OOS warning
    assert any("pseudo-out-of-sample" in w for w in warns)


def test_apply_combinations_idempotent():
    master = _master_ragged()
    master["combined"] = False
    spec = _spec_stub()
    spec = dataclasses.replace(spec, combinations=(CombinationContender("POOL", "mean"),))
    once = apply_combinations(master, spec)
    twice = apply_combinations(once, spec)
    # POOL must appear exactly once, not be re-combined into itself
    assert (once["contender"] == "POOL").sum() == (twice["contender"] == "POOL").sum()
