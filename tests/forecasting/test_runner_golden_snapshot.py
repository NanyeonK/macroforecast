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
    # other input_kinds -- deterministic representatives so the refactor cannot silently
    # break the target / volatility routing. (Panel routing is guarded separately in
    # test_panel_routing_intact because the var/panel path is not bit-reproducible, see
    # the note there -- so its forecasts cannot be pinned to exact values.)
    ("direct", "naive", {}, [2, 3]),
    ("path_average", "naive", {}, [2, 3]),
    ("direct", "arima", {}, [2]),
    ("direct", "garch11", {}, [2]),
]

_KEY_COLS = ["forecast_policy", "model", "horizon", "origin", "date", "prediction", "actual"]

# Models in _MATRIX whose backing package is an OPTIONAL extra (not part of the
# core install). CI core installs only ".[ci]" (no such extras), so these cells
# cannot run there: the runner yields an empty frame for the arm and the row
# count silently shrinks below the committed golden (18 origins x 1 horizon per
# cell). Both the snapshot run and the golden comparison exclude the models
# whose import is unavailable in the CURRENT environment; regeneration
# (MF_UPDATE_GOLDEN=1) refuses to write a partial fixture so the committed
# golden always carries the full matrix.
_OPTIONAL_MODEL_IMPORTS = {"garch11": "arch"}


def _missing_optional_models() -> "set[str]":
    import importlib.util

    return {
        model
        for model, package in _OPTIONAL_MODEL_IMPORTS.items()
        if importlib.util.find_spec(package) is None
    }


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
    from macroforecast.models.specs import MODEL_SPECS

    frames = []
    missing = _missing_optional_models()
    for policy, model, params, horizons in _MATRIX:
        if model in missing:
            continue
        input_kind = MODEL_SPECS[model].input_kind
        if input_kind == "panel":
            # panel-input models consume the panel directly (a separate strategy).
            feats, target_transform = None, "level"
        elif policy == "recursive":
            feats, target_transform = feats_ar, "level"
        else:
            feats, target_transform = feats_exog, "value"
        spec = pipeline_spec(
            data=bundle,
            targets=[TargetSpec(name="Y", transform=target_transform, policy=policy)],
            horizons=horizons, window=win,
            arms=[Arm(name="M", model=model, features=feats, is_benchmark=True, params=params)],
            evaluation=EvalSpec(benchmark="M", metrics=("rmse",)), n_jobs=1,
            on_unsupported_direct="warn",
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
    missing = _missing_optional_models()
    current = _snapshot()
    if os.environ.get("MF_UPDATE_GOLDEN") == "1":
        assert not missing, (
            f"refusing to regenerate a PARTIAL golden fixture: optional model "
            f"package(s) unavailable for {sorted(missing)}; install the missing "
            "extras so the committed fixture carries the full matrix"
        )
        _FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        current.to_parquet(_FIXTURE)
        return  # regeneration mode: no assertion
    assert _FIXTURE.exists(), (
        f"golden fixture missing at {_FIXTURE}; generate it with "
        "MF_UPDATE_GOLDEN=1 pytest tests/forecasting/test_runner_golden_snapshot.py"
    )
    golden = pd.read_parquet(_FIXTURE)
    if missing:
        golden = golden[~golden["model"].isin(missing)].reset_index(drop=True)
    assert list(current.columns) == list(golden.columns)
    assert len(current) == len(golden), (
        f"row count changed: {len(current)} vs golden {len(golden)}"
        + (f" (models excluded as unavailable here: {sorted(missing)})" if missing else "")
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


def test_panel_routing_intact():
    """Guard that the panel-input strategy still routes, forecasts, and labels correctly.

    var/panel IS deterministic (pure OLS). Issue #423 (panel test window included the
    origin itself + ``_panel_prediction_horizon`` floored to max(1, ...)) used to mean a
    horizons=[2] request emitted a multi-step path but tagged every row horizon=1, so
    (horizon, origin) was not a unique key and the rows could not be pinned by
    (policy, model, horizon, origin) like the golden snapshot. That fix has landed
    (WindowSpec.origins(exclude_origin=True) for the panel call site + the unfloored
    positional-distance helper), so this guard now also pins the horizon label and
    date offset -- not just routing survival -- for both policies.
    """
    bundle = _dataset()
    win = mf.window.from_cutoffs(
        test_start="2008-01-01", test_end="2009-06-01", mode="expanding",
        val_method="last_block", retrain_every=1,
    )
    for policy in ("direct", "path_average"):
        spec = pipeline_spec(
            data=bundle, targets=[TargetSpec(name="Y", transform="level", policy=policy)],
            horizons=[2], window=win,
            arms=[Arm(name="M", model="var", features=None, is_benchmark=True)],
            evaluation=EvalSpec(benchmark="M", metrics=("rmse",)), n_jobs=1,
        )
        f = run_pipeline(spec).forecasts
        assert not f.empty, f"var/{policy} produced no forecasts -- panel routing broke"
        assert f["prediction"].notna().any(), f"var/{policy} produced no predictions"
        # Post-#423: every emitted row must be tagged with the REQUESTED horizon (2),
        # not the pre-fix constant 1, and (horizon, origin) must be a unique key.
        assert sorted(f["horizon"].unique().tolist()) == [2], (
            f"var/{policy} horizon labels drifted from the requested horizon: "
            f"{sorted(f['horizon'].unique().tolist())}"
        )
        assert not f.duplicated(["horizon", "origin"]).any(), (
            f"var/{policy} produced duplicate (horizon, origin) keys"
        )
        # date must be exactly origin + 2 positional steps in the panel index.
        positions = bundle.panel.index.get_indexer(f["origin"])
        assert (positions >= 0).all(), f"var/{policy}: every origin must be a real panel date"
        expected_dates = bundle.panel.index[positions + 2]
        assert (f["date"].to_numpy() == expected_dates.to_numpy()).all(), (
            f"var/{policy} date != origin + horizon"
        )
