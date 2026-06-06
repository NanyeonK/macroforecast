"""Stage 2: pipeline evaluation (accuracy, significance, MCS, combinations)."""
import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import (
    Arm, CombinationContender, EvalSpec, apply_combinations, evaluate, pipeline_spec, run_arms,
)


def _bundle(n=96):
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, n)
    frame = pd.DataFrame({"y": 1.0 + 2.0 * x + rng.standard_normal(n) * 0.05, "x1": x}, index=idx)
    return mf.data.custom_dataset(frame, transform_codes={"y": 1, "x1": 1})


def _window():
    return mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=36),
        val=mf.window.val_last_block(size=12),
        test=mf.window.test_origins(horizon=1, step=3),
    )


def _spec(**over):
    feats = mf.feature_engineering.feature_spec(target="y", predictors=["x1"], lags=1, target_lags=(0, 1))
    kw = dict(
        data=_bundle(), targets=["y"], horizons=[1], window=_window(),
        arms=[Arm("AR", model="ar", features=feats),
              Arm("OLS", model="ols", features=feats, nested_in_benchmark=True),
              Arm("RIDGE", model="ridge", features=feats, nested_in_benchmark=True)],
        evaluation=EvalSpec(benchmark="AR"), save_models=False,
    )
    kw.update(over)
    return pipeline_spec(**kw)


def test_accuracy_relative_to_benchmark():
    spec = _spec()
    master = run_arms(spec)
    res = evaluate(master, spec)
    acc = res["accuracy"]
    assert {"contender", "rmse", "relative_mse", "r2_oos", "is_benchmark"} <= set(acc.columns)
    # benchmark relative_mse == 1 and r2_oos == 0
    bench_row = acc[acc["is_benchmark"]].iloc[0]
    assert np.isclose(bench_row["relative_mse"], 1.0)
    assert np.isclose(bench_row["r2_oos"], 0.0)


def test_significance_has_dm_and_cw():
    spec = _spec()
    res = evaluate(run_arms(spec), spec)
    sig = res["significance"]
    assert not sig.empty
    assert {"dm_p", "cw_p"}.issubset(sig.columns)
    # benchmark itself is not in the significance table
    assert "AR" not in set(sig["contender"])


def test_combination_added_as_contender():
    spec = _spec(combinations=[CombinationContender(name="POOL", method="mean")])
    master = run_arms(spec)
    full = apply_combinations(master, spec)
    assert "POOL" in set(full["contender"])
    # POOL evaluated alongside the rest
    res = evaluate(master, spec)
    assert "POOL" in set(res["accuracy"]["contender"])


def test_bates_granger_combination_contender():
    spec = _spec(combinations=[CombinationContender(name="BG", method="bates_granger", params={"min_periods": 8})])
    full = apply_combinations(run_arms(spec), spec)
    bg = full[full["contender"] == "BG"]
    assert not bg.empty
    assert np.isfinite(bg["prediction"].dropna().to_numpy()).all()


def test_mcs_runs_when_enough_origins():
    spec = _spec()
    res = evaluate(run_arms(spec), spec)
    mcs = res["mcs"]
    # may be empty on tiny samples, but if present must have in_mcs flags
    if not mcs.empty:
        assert "in_mcs" in mcs.columns
        assert mcs["in_mcs"].any()
