"""An arm may carry its own window, run independently of the shared spec window.

This lets, for example, the autoregression run on a window with no validation
block while the other arms use the shared validated window, and it makes each
cell of the pipeline independently configurable.
"""
import numpy as np
import pandas as pd

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, pipeline_spec, run_pipeline


def _bundle(n=120):
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, n)
    frame = pd.DataFrame(
        {"y": 1.0 + 2.0 * x + rng.standard_normal(n) * 0.05, "x1": x}, index=idx
    )
    return mf.data.custom_dataset(frame, transform_codes={"y": 1, "x1": 1})


def _feats():
    return mf.feature_engineering.feature_spec(
        target="y", predictors=["x1"], lags=1, target_lags=(0, 1)
    )


def test_arm_window_overrides_shared_window() -> None:
    shared = mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=36),
        val=mf.window.val_last_block(size=12),
        test=mf.window.test_origins(horizon=1, step=6),  # every 6th origin
    )
    # the AR arm forecasts at EVERY origin via its own window (step 1, no val)
    ar_window = mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=36),
        test=mf.window.test_origins(horizon=1, step=1),
    )
    spec = pipeline_spec(
        data=_bundle(),
        targets=["y"],
        horizons=[1],
        window=shared,
        arms=[
            Arm("AR", model="ar", features=_feats(), window=ar_window),
            Arm("OLS", model="ols", features=_feats()),
        ],
        evaluation=EvalSpec(benchmark="OLS"),
        save_models=False,
    )
    report = run_pipeline(spec)
    fc = report.forecasts
    n_ar = int((fc["arm"] == "AR").sum())
    n_ols = int((fc["arm"] == "OLS").sum())
    # the AR arm steps every origin, the OLS arm every sixth -> strictly more AR rows
    assert n_ar > n_ols, f"expected per-arm window to give AR more origins, got AR={n_ar} OLS={n_ols}"


def test_arm_without_window_uses_shared() -> None:
    shared = mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=36),
        val=mf.window.val_last_block(size=12),
        test=mf.window.test_origins(horizon=1, step=6),
    )
    spec = pipeline_spec(
        data=_bundle(),
        targets=["y"],
        horizons=[1],
        window=shared,
        arms=[Arm("AR", model="ar", features=_feats()),
              Arm("OLS", model="ols", features=_feats())],
        evaluation=EvalSpec(benchmark="OLS"),
        save_models=False,
    )
    report = run_pipeline(spec)
    fc = report.forecasts
    # both arms share the window -> same number of origins
    assert int((fc["arm"] == "AR").sum()) == int((fc["arm"] == "OLS").sum())
