"""Stage 1: run_arms executes arms into the master forecast frame."""
import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import (
    Arm, EvalSpec, pipeline_spec, run_arms, run_pipeline,
)


def _bundle(n=72):
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, n)
    frame = pd.DataFrame(
        {"y": 1.0 + 2.0 * x + rng.standard_normal(n) * 0.05, "x1": x},
        index=idx,
    )
    # y is tcode 1 (level / stationary here) to keep the target simple
    return mf.data.custom_dataset(frame, transform_codes={"y": 1, "x1": 1})


def _window():
    return mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=36),
        val=mf.window.val_last_block(size=12),
        test=mf.window.test_origins(horizon=1, step=6),
    )


def _spec(**over):
    feats = mf.feature_engineering.feature_spec(target="y", predictors=["x1"], lags=1, target_lags=(0, 1))
    kw = dict(
        data=_bundle(), targets=["y"], horizons=[1, 3], window=_window(),
        arms=[Arm("AR", model="ar", features=feats), Arm("OLS", model="ols", features=feats)],
        evaluation=EvalSpec(benchmark="AR"), save_models=False,
    )
    kw.update(over)
    return pipeline_spec(**kw)


def test_master_frame_has_required_columns_and_tags():
    master = run_arms(_spec())
    assert not master.empty
    for col in ("arm", "contender", "model", "target", "horizon", "prediction", "actual"):
        assert col in master.columns
    # both arms present as contenders (single model -> contender == arm name)
    assert set(master["arm"]) == {"AR", "OLS"}
    assert set(master["contender"]) == {"AR", "OLS"}
    assert set(master["horizon"]) == {1, 3}


def test_comparing_models_uses_one_arm_per_model():
    # Comparing models = multiple Arms identical except for ``model``; each arm is
    # its own contender (no arm:model labels).
    feats = mf.feature_engineering.feature_spec(target="y", predictors=["x1"], lags=1, target_lags=(0, 1))
    spec = _spec(
        arms=[
            Arm("bench", model="ar", features=feats),
            Arm("OLS", model="ols", features=feats),
            Arm("RIDGE", model="ridge", features=feats),
        ],
        evaluation=EvalSpec(benchmark="bench"),
    )
    master = run_arms(spec)
    assert set(master["arm"]) == {"bench", "OLS", "RIDGE"}
    # contender == arm name for every row
    assert (master["contender"] == master["arm"]).all()


def test_predictions_are_finite_on_each_contender():
    master = run_arms(_spec())
    assert master["prediction"].notna().any()
    assert np.isfinite(master["prediction"].dropna().to_numpy()).all()


def _sorted_master(frame):
    cols = ["arm", "contender", "target", "horizon", "origin", "date"]
    cols = [c for c in cols if c in frame.columns]
    return frame.sort_values(cols).reset_index(drop=True)


def test_checkpoint_dir_threads_through_and_creates_per_cell_layout(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    spec = _spec(checkpoint_dir=str(ckpt_dir))
    assert spec.checkpoint_dir == str(ckpt_dir)

    master = run_arms(spec)
    assert not master.empty

    # Layout: <dir>/<target>__<arm>/h<h>/origin_<pos>.parquet (multi-horizon spec).
    cells = sorted(p.name for p in ckpt_dir.iterdir())
    assert cells == ["y__AR", "y__OLS"]
    for cell in cells:
        horizon_dirs = sorted(p.name for p in (ckpt_dir / cell).iterdir())
        assert horizon_dirs == ["h1", "h3"]
        for hd in horizon_dirs:
            files = list((ckpt_dir / cell / hd).glob("origin_*.parquet"))
            assert files


def _exploding_fit(X, y):
    raise RuntimeError("boom: this arm fails on purpose")


def test_failed_cell_is_isolated_and_recorded():
    # One arm raises during run(); the managed run still returns the other arms'
    # forecasts and records the failure (cell identity + error) on the report.
    feats = mf.feature_engineering.feature_spec(
        target="y", predictors=["x1"], lags=1, target_lags=(0, 1)
    )
    boom = mf.models.custom_model("boom", _exploding_fit)
    spec = _spec(
        arms=[
            Arm("AR", model="ar", features=feats),
            Arm("BOOM", model=boom, features=feats),
            Arm("OLS", model="ols", features=feats),
        ],
        evaluation=EvalSpec(benchmark="AR"),
    )

    report = run_pipeline(spec)

    # The healthy arms are present; the failing arm is absent from the master frame.
    assert set(report.forecasts["arm"]) == {"AR", "OLS"}

    # The failure is recorded with the cell identity and the error text.
    assert len(report.failed_cells) >= 1
    failed_arms = {fc["arm"] for fc in report.failed_cells}
    assert failed_arms == {"BOOM"}
    assert all("boom" in fc["error"].lower() for fc in report.failed_cells)
    assert all("horizons" in fc and "target" in fc for fc in report.failed_cells)
    # Mirrored into the leakage audit too.
    assert report.leakage_audit["failed_cells"] == list(report.failed_cells)


def test_failed_cell_isolation_is_identical_under_parallel():
    # The same isolation holds when cells are split per-horizon across a pool.
    feats = mf.feature_engineering.feature_spec(
        target="y", predictors=["x1"], lags=1, target_lags=(0, 1)
    )
    boom = mf.models.custom_model("boom", _exploding_fit)
    spec = _spec(
        arms=[
            Arm("AR", model="ar", features=feats),
            Arm("BOOM", model=boom, features=feats),
            Arm("OLS", model="ols", features=feats),
        ],
        evaluation=EvalSpec(benchmark="AR"),
        n_jobs=2,
    )

    report = run_pipeline(spec)
    assert set(report.forecasts["arm"]) == {"AR", "OLS"}
    assert {fc["arm"] for fc in report.failed_cells} == {"BOOM"}
    # Parallel splits BOOM into one failed cell per horizon (h1, h3).
    assert len(report.failed_cells) == 2


def test_checkpoint_dir_resume_matches_no_checkpoint(tmp_path):
    baseline = _sorted_master(run_arms(_spec()))

    ckpt_dir = tmp_path / "ckpt"
    # First pass writes the checkpoint; second pass resumes entirely from disk.
    run_arms(_spec(checkpoint_dir=str(ckpt_dir)))
    resumed = _sorted_master(run_arms(_spec(checkpoint_dir=str(ckpt_dir))))

    assert list(resumed["arm"]) == list(baseline["arm"])
    assert list(resumed["horizon"]) == list(baseline["horizon"])
    np.testing.assert_allclose(
        resumed["prediction"].to_numpy(dtype=float),
        baseline["prediction"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-9,
    )
