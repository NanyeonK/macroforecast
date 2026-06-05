"""Stage 3: deferred ML interpretation (interpret_pipeline)."""
import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import (
    Arm, EvalSpec, InterpretSpec, interpret_pipeline, pipeline_spec, run_pipeline,
)


def _bundle(n=120):
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    rng = np.random.default_rng(0)
    x1 = rng.standard_normal(n).cumsum()
    x2 = rng.standard_normal(n)
    y = 0.8 * x1 + 0.1 * x2 + rng.standard_normal(n) * 0.2
    frame = pd.DataFrame({"y": y, "x1": x1, "x2": x2}, index=idx)
    return mf.data.custom_dataset(frame, transform_codes={"y": 1, "x1": 1, "x2": 1})


def _spec(interpret=None, **over):
    feats = mf.feature_engineering.feature_spec(target="y", predictors=["x1", "x2"], lags=1, target_lags=(0, 1))
    w = mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=48),
        val=mf.window.val_last_block(size=12),
        test=mf.window.test_origins(horizon=1, step=6),
    )
    kw = dict(
        data=_bundle(), targets=[mf.pipeline.TargetSpec("y", transform="level")], horizons=[1], window=w,
        arms=[Arm("AR", model="ar", features=feats),
              Arm("RF", model="random_forest", features=feats, interpret=interpret)],
        evaluation=EvalSpec(benchmark="AR"), save_models=False,
    )
    kw.update(over)
    return pipeline_spec(**kw)


def test_interpret_pipeline_shap_importance():
    spec = _spec(interpret=InterpretSpec(methods=("shap",)))
    report = run_pipeline(spec)
    out = interpret_pipeline(report)
    assert "RF" in out
    # AR has no interpret spec -> skipped
    assert "AR" not in out
    rf = out["RF"]
    table = next(iter(rf.values()))["shap"]
    assert isinstance(table, pd.DataFrame) and not table.empty
    # stored on the report
    assert report.interpretation is out


def test_methods_override_and_multiple_at_once():
    spec = _spec(interpret=InterpretSpec(methods=("shap",)))
    report = run_pipeline(spec)
    out = interpret_pipeline(report, methods=("shap", "ale"), arms=("RF",))
    rf_model = next(iter(out["RF"].values()))
    assert set(rf_model.keys()) == {"shap", "ale"}


def test_no_interpret_arm_returns_empty():
    spec = _spec(interpret=None)
    report = run_pipeline(spec)
    out = interpret_pipeline(report)
    assert out == {}
