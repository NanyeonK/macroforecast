"""Golden-master snapshot of the runner across policies -- the refactor safety net.

Phase 0 of the runner.py policy-strategy decomposition. This pins the EXACT forecasts
the runner produces today, across a representative matrix of (forecast_policy x model x
horizon), into a committed fixture. Every behavior-preserving refactor step must keep
this byte-identical (atol 1e-10). It complements the targeted oracles (h1 direct==path,
serial==parallel, path/far/AL ground-truth) with one integration-level guard that would
catch any drift they miss.

To regenerate the fixture after an INTENTIONAL behavior change, run:
    MF_UPDATE_GOLDEN=1 pytest tests/forecasting/test_runner_golden_snapshot.py
and review the diff deliberately -- a change here means forecasts moved.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline

_FIXTURE = Path(__file__).parent / "_golden" / "runner_snapshot.parquet"

# (policy, model, params, horizons). Chosen to exercise every feature-matrix policy
# path and the shared select->fit->predict->record skeleton across single-shot,
# information-criterion, and CV-selected models.
_MATRIX = [
    ("direct", "ols", {}, [1, 2, 3]),
    ("direct", "ar", {}, [1, 2, 3]),
    ("direct", "far", {"n_factors": 3, "n_lag": 2}, [1, 2, 3]),
    ("direct", "elastic_net", {}, [1, 2]),
    ("direct_average", "ar", {}, [2, 3]),
    ("direct_average", "far", {"n_factors": 3, "n_lag": 2}, [2, 3]),
    ("path_average", "ols", {}, [2, 3]),
    ("path_average", "ar", {}, [2, 3]),
    ("path_average", "far", {"n_factors": 3, "n_lag": 2}, [2, 3]),
    ("recursive", "ols", {}, [2]),
    ("recursive", "ridge", {}, [2]),
]

_KEY_COLS = ["forecast_policy", "model", "horizon", "origin", "date", "prediction", "actual"]


def _dataset():
    """A fixed, mildly factor-structured monthly panel (deterministic)."""
    n = 260
    rng = np.random.default_rng(20260703)
    # two latent factors driving the predictor block + idiosyncratic noise
    f = np.cumsum(rng.normal(size=(n, 2)) * 0.3, axis=0)
    load = rng.normal(size=(6, 2))
    predictors = f @ load.T + rng.normal(size=(n, 6)) * 0.2
    y = 0.02 * f[:, 0] - 0.015 * f[:, 1] + rng.normal(size=n) * 0.05
    idx = pd.date_range("1990-01-01", periods=n, freq="MS")
    cols = {f"x{i}": predictors[:, i] for i in range(6)}
    cols["Y"] = y
    panel = pd.DataFrame(cols, index=idx)
    panel.index.name = "date"
    return mf.data.custom_dataset(panel, transform_codes={c: 1 for c in panel.columns})


def _snapshot():
    """Run the whole matrix and return one canonical, sorted forecasts frame."""
    bundle = _dataset()
    win = mf.window.from_cutoffs(
        test_start="2008-01-01", test_end="2009-06-01", mode="expanding",
        val_method="last_block", retrain_every=1,
    )
    # Feature-matrix policies use exogenous predictors; the recursive policy is
    # autoregressive-only (it rejects exogenous predictors and reads its AR inputs
    # from target_lags), so it gets its own spec.
    feats_exog = mf.feature_engineering.feature_spec(
        target="Y", predictors="all", lags=range(1, 3),
        target_lags=range(0, 4), target_transform="value",
    )
    # recursive supports level/change/growth targets, not 'value'; use a level target.
    feats_ar = mf.feature_engineering.feature_spec(
        target="Y", predictors=[], lags=None,
        target_lags=range(0, 4), target_transform="level",
    )
    frames = []
    for policy, model, params, horizons in _MATRIX:
        feats = feats_ar if policy == "recursive" else feats_exog
        target_transform = "level" if policy == "recursive" else "value"
        spec = pipeline_spec(
            data=bundle,
            targets=[TargetSpec(name="Y", transform=target_transform, policy=policy)],
            horizons=horizons, window=win,
            arms=[Arm(name="M", model=model, features=feats, is_benchmark=True, params=params)],
            evaluation=EvalSpec(benchmark="M", metrics=("rmse",)), n_jobs=1,
        )
        f = run_pipeline(spec).forecasts
        if f.empty:
            continue
        f = f.copy()
        f["model"] = model  # pin the intended model label independent of arm name
        frames.append(f[[c for c in _KEY_COLS if c in f.columns]])
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["forecast_policy", "model", "horizon", "origin"]).reset_index(drop=True)
    # normalise dtypes for a stable on-disk representation
    out["origin"] = out["origin"].astype("datetime64[ns]")
    out["date"] = out["date"].astype("datetime64[ns]")
    return out


def test_runner_matrix_matches_golden_snapshot():
    current = _snapshot()
    if os.environ.get("MF_UPDATE_GOLDEN") == "1":
        _FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        current.to_parquet(_FIXTURE)
        return  # regeneration mode: no assertion
    assert _FIXTURE.exists(), (
        f"golden fixture missing at {_FIXTURE}; generate it with "
        "MF_UPDATE_GOLDEN=1 pytest tests/forecasting/test_runner_golden_snapshot.py"
    )
    golden = pd.read_parquet(_FIXTURE)
    assert list(current.columns) == list(golden.columns)
    assert len(current) == len(golden), (
        f"row count changed: {len(current)} vs golden {len(golden)}"
    )
    # exact on labels/dates, tight tolerance on the float forecasts
    pd.testing.assert_frame_equal(
        current.drop(columns=["prediction", "actual"]),
        golden.drop(columns=["prediction", "actual"]),
        check_dtype=False,
    )
    for col in ("prediction", "actual"):
        np.testing.assert_allclose(
            current[col].to_numpy(dtype=float),
            golden[col].to_numpy(dtype=float),
            rtol=0.0, atol=1e-10, equal_nan=True,
            err_msg=f"{col} drifted from the golden snapshot",
        )
