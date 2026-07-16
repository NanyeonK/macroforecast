"""significance_table must not crash when a Diebold-Mariano / Clark-West statistic is
None -- which happens for a DEGENERATE pair, most commonly two IDENTICAL forecasts
(zero loss differential -> undefined test). The degenerate pair is recorded as NaN and
every other (well-posed) pair is still evaluated.

Regression: previously ``float(cast(float, dm.statistic))`` raised
``TypeError: float() argument must be ... not 'NoneType'``; ``run_pipeline`` swallowed
it and left ``report.accuracy`` empty, blocking native DM/MCS (Table A1).
"""
import math

import numpy as np
import pandas as pd

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, pipeline_spec
from macroforecast.pipeline.evaluate import significance_table


def _spec(benchmark="AR"):
    feats = mf.feature_engineering.feature_spec(target="y", predictors=["x1"], lags=1, target_lags=(0, 1))
    idx = pd.date_range("2000-01-31", periods=96, freq="ME", name="date")
    x = np.linspace(0.0, 1.0, 96)
    frame = pd.DataFrame({"y": 1.0 + 2.0 * x, "x1": x}, index=idx)
    bundle = mf.data.custom_dataset(frame, transform_codes={"y": 1, "x1": 1})
    w = mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=36),
        val=mf.window.val_last_block(size=12),
        test=mf.window.test_origins(horizon=1, step=3),
    )
    return pipeline_spec(
        data=bundle, targets=["y"], horizons=[1], window=w,
        arms=[Arm("AR", model="ar", features=feats),
              Arm("DUP", model="ar", features=feats),
              Arm("DIFF", model="ols", features=feats)],
        evaluation=EvalSpec(benchmark=benchmark, tests=("dm",)), save_models=False,
    )


def _master():
    idx = pd.date_range("2000-01-31", periods=12, freq="ME")
    actual = np.linspace(1.0, 2.0, 12)
    preds = {
        "AR": actual + 0.10,          # benchmark
        "DUP": actual + 0.10,         # IDENTICAL to benchmark -> DM None -> NaN
        "DIFF": actual + 0.02 * np.sin(np.arange(12)),  # different -> finite DM
    }
    rows = []
    for contender, p in preds.items():
        for dt, pr, ac in zip(idx, p, actual):
            rows.append({"target": "y", "horizon": 1, "origin": dt,
                         "contender": contender, "prediction": pr, "actual": ac})
    return pd.DataFrame(rows)


def test_significance_table_does_not_crash_on_identical_forecasts():
    spec = _spec()
    sig = significance_table(_master(), spec)  # must NOT raise
    assert isinstance(sig, pd.DataFrame) and not sig.empty
    assert {"target", "horizon", "contender", "dm_stat", "dm_p"}.issubset(sig.columns)

    by = sig.set_index("contender")
    # identical-forecast pair -> DM undefined -> recorded as NaN (not-significant)
    assert "DUP" in by.index
    assert math.isnan(float(by.loc["DUP", "dm_stat"]))
    assert math.isnan(float(by.loc["DUP", "dm_p"]))
    # well-posed pair still evaluated with a finite statistic
    assert "DIFF" in by.index
    assert math.isfinite(float(by.loc["DIFF", "dm_stat"]))
