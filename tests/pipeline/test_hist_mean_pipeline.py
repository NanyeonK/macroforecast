from __future__ import annotations

import numpy as np
import pandas as pd

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline


def _bundle(n: int = 156) -> mf.DataBundle:
    idx = pd.date_range("1990-01-31", periods=n, freq="ME", name="date")
    rng = np.random.default_rng(11)
    y = np.zeros(n, dtype=float)
    shock = rng.normal(scale=0.25, size=n)
    for i in range(1, n):
        y[i] = 0.75 * y[i - 1] + 0.02 * i + shock[i]
    panel = pd.DataFrame({"Y": y}, index=idx)
    return mf.data.custom_dataset(panel, transform_codes={"Y": 1})


def _window() -> mf.window.WindowSpec:
    return mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=72),
        val=mf.window.val_last_block(size=12),
        test=mf.window.test_origins(horizon=1, step=3),
    )


def test_hist_mean_benchmark_supports_direct_cw_pipeline_smoke() -> None:
    features = mf.feature_engineering.feature_spec(
        target="Y",
        predictors=[],
        lags=None,
        target_lags=(0, 1),
    )
    spec = pipeline_spec(
        data=_bundle(),
        targets=[TargetSpec("Y", transform="level", policy="direct")],
        horizons=[1],
        window=_window(),
        arms=[
            Arm("HA", model="hist_mean", features=features, is_benchmark=True),
            Arm("AR", model="ar", features=features, nested_in_benchmark=True),
        ],
        evaluation=EvalSpec(benchmark="HA", metrics=("rmse",), tests=("cw",)),
        save_models=False,
    )

    report = run_pipeline(spec)

    assert set(report.forecasts["contender"]) == {"HA", "AR"}
    assert set(report.forecasts["model_spec"]) == {"hist_mean", "ar"}
    sig = report.significance
    assert not sig.empty
    assert set(sig["contender"]) == {"AR"}
    assert "cw_p" in sig.columns
    assert sig["cw_p"].notna().all()
