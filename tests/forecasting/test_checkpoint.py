"""Runner-level incremental checkpoint tests.

These cover the feature-matrix single-horizon execution path (direct,
direct_average, path_average), which is what the POOS replications use. The
checkpoint persists a LEAN forecast record per origin and resumes by skipping
origins already on disk.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.forecasting import checkpoint as ckpt


def teardown_function() -> None:
    mf.meta.reset_config()


def _panel(n: int = 48) -> pd.DataFrame:
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    x = np.linspace(0.0, 1.0, n)
    return pd.DataFrame(
        {
            "y": 1.0 + 2.0 * x + 0.1 * np.sin(np.arange(n) / 2.0),
            "x1": x,
            "x2": np.sin(np.arange(n) / 3.0),
        },
        index=idx,
    )


def _window() -> mf.window.WindowSpec:
    return mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=24),
        val=mf.window.val_last_block(size=8),
        test=mf.window.test_origins(horizon=1, step=6),
    )


def _features() -> mf.feature_engineering.FeatureSpec:
    return mf.feature_engineering.feature_spec(target="y", target_lags=[1, 2])


def _run(checkpoint_path=None, policy: str = "direct"):
    return mf.forecasting.run(
        _panel(),
        "ols",
        window=_window(),
        features=_features(),
        forecast_policy=policy,
        target_transform="change" if policy != "direct" else None,
        horizon=3 if policy != "direct" else 1,
        save_models=False,
        checkpoint_path=checkpoint_path,
    )


def _cell_horizon_dir(cell: Path, policy: str = "direct") -> Path:
    """Resolve the per-horizon checkpoint subdirectory ``run()`` writes into.

    The runner namespaces each cell's checkpoint by horizon (``cell/h<h>``) so a
    per-cell directory reused across horizons never collides. ``_run`` uses
    horizon 1 for the direct policy and horizon 3 otherwise.
    """
    return cell / ("h1" if policy == "direct" else "h3")


def _sorted_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cols = ["origin_pos", "model", "horizon", "date"]
    return (
        frame.sort_values(cols)
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# 1. checkpointing does not change the forecasts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("policy", ["direct", "direct_average", "path_average"])
def test_checkpoint_produces_identical_forecasts(tmp_path: Path, policy: str) -> None:
    baseline = _run(checkpoint_path=None, policy=policy)
    checked = _run(checkpoint_path=tmp_path / "cell", policy=policy)

    base = _sorted_frame(baseline.to_frame())
    chk = _sorted_frame(checked.to_frame())

    assert list(base["origin_pos"]) == list(chk["origin_pos"])
    np.testing.assert_allclose(
        base["prediction"].to_numpy(dtype=float),
        chk["prediction"].to_numpy(dtype=float),
        rtol=0,
        atol=0,
    )
    # actuals carried through identically too
    np.testing.assert_allclose(
        base["actual"].to_numpy(dtype=float),
        chk["actual"].to_numpy(dtype=float),
        rtol=0,
        atol=0,
    )


# --------------------------------------------------------------------------- #
# 2. resume skips origins already on disk
# --------------------------------------------------------------------------- #
def test_resume_skips_completed_origins(tmp_path: Path, monkeypatch) -> None:
    cell = tmp_path / "cell"

    # First, run fully to get the ground-truth complete frame and the per-origin
    # files written.
    full = _run(checkpoint_path=cell, policy="direct")
    full_frame = _sorted_frame(full.to_frame())

    hdir = _cell_horizon_dir(cell, "direct")
    origin_files = sorted(hdir.glob("origin_*.parquet"))
    assert len(origin_files) >= 2, "need multiple origins to test resume"

    # Remove the LAST origin's file so a resume must recompute exactly one origin.
    completed_before = ckpt.completed_origin_positions(hdir)
    last_pos = max(completed_before)
    (hdir / f"origin_{last_pos}.parquet").unlink()
    remaining = ckpt.completed_origin_positions(hdir)
    assert last_pos not in remaining

    # Spy on the per-origin fit/predict to record which origins are actually
    # COMPUTED on resume.
    import macroforecast.forecasting.runner as runner

    computed_origins: list[int] = []
    original = runner._fit_predict_origin

    def _spy(item, *args, **kwargs):
        pos = item["row"].get("origin_pos")
        if pos is not None:
            computed_origins.append(int(pos))
        return original(item, *args, **kwargs)

    monkeypatch.setattr(runner, "_fit_predict_origin", _spy)

    resumed = _run(checkpoint_path=cell, policy="direct")
    resumed_frame = _sorted_frame(resumed.to_frame())

    # Only the deleted origin should have been recomputed; the others were skipped.
    assert computed_origins == [last_pos]
    assert set(computed_origins).isdisjoint(remaining)

    # The resumed frame must still contain ALL origins with the original values.
    assert list(resumed_frame["origin_pos"]) == list(full_frame["origin_pos"])
    np.testing.assert_allclose(
        resumed_frame["prediction"].to_numpy(dtype=float),
        full_frame["prediction"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-12,
    )


# --------------------------------------------------------------------------- #
# 3. checkpoint files are scalar-only (parquet serialisable)
# --------------------------------------------------------------------------- #
def test_checkpoint_files_are_scalar_only(tmp_path: Path) -> None:
    cell = tmp_path / "cell"
    _run(checkpoint_path=cell, policy="direct_average")

    files = sorted(_cell_horizon_dir(cell, "direct_average").glob("origin_*.parquet"))
    assert files, "expected per-origin parquet files"

    for path in files:
        frame = pd.read_parquet(path)
        assert list(frame.columns) == list(ckpt.LEAN_FORECAST_COLUMNS)
        for column in frame.columns:
            for value in frame[column].tolist():
                # No dict / list / struct columns: only scalars (or None/NaT).
                assert not isinstance(value, (dict, list, tuple, set)), (
                    f"non-scalar value {value!r} in column {column!r}"
                )


# --------------------------------------------------------------------------- #
# 3b. when an in-loop stage carries fitted state forward (feature update='never'),
#     resume must still recompute every origin (to rebuild the once-fitted state)
#     but skip the redundant WRITE, and the merged frame stays complete + correct.
# --------------------------------------------------------------------------- #
def test_resume_recomputes_when_stage_state_carries_forward(
    tmp_path: Path, monkeypatch
) -> None:
    cell = tmp_path / "cell"
    fpol = mf.window.stage_policy("fit_window", update="never")

    def go(cp):
        return mf.forecasting.run(
            _panel(),
            "ols",
            window=_window(),
            features=_features(),
            feature_policy=fpol,
            save_models=False,
            checkpoint_path=cp,
        )

    full = go(cell)
    full_frame = _sorted_frame(full.to_frame())

    hdir = _cell_horizon_dir(cell, "direct")  # go() uses the default horizon 1
    all_pos = sorted(ckpt.completed_origin_positions(hdir))
    assert len(all_pos) >= 2
    last_pos = all_pos[-1]
    (hdir / f"origin_{last_pos}.parquet").unlink()

    import macroforecast.forecasting.runner as runner

    computed: list[int] = []
    original = runner._fit_predict_origin

    def _spy(item, *args, **kwargs):
        computed.append(int(item["row"].get("origin_pos")))
        return original(item, *args, **kwargs)

    monkeypatch.setattr(runner, "_fit_predict_origin", _spy)

    resumed = go(cell)
    resumed_frame = _sorted_frame(resumed.to_frame())

    # Every origin recomputed (state carries forward), not just the deleted one.
    assert sorted(computed) == all_pos
    # ... but only the deleted origin's file is (re)written.
    assert ckpt.completed_origin_positions(hdir) == set(all_pos)
    # ... and the merged frame is complete and numerically identical.
    assert list(resumed_frame["origin_pos"]) == list(full_frame["origin_pos"])
    np.testing.assert_allclose(
        resumed_frame["prediction"].to_numpy(dtype=float),
        full_frame["prediction"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-12,
    )


# --------------------------------------------------------------------------- #
# 4. resume returns the complete frame even when ALL origins are on disk
# --------------------------------------------------------------------------- #
def test_resume_with_all_origins_on_disk(tmp_path: Path, monkeypatch) -> None:
    cell = tmp_path / "cell"
    full = _run(checkpoint_path=cell, policy="direct")
    full_frame = _sorted_frame(full.to_frame())

    import macroforecast.forecasting.runner as runner

    computed: list[int] = []
    original = runner._fit_predict_origin

    def _spy(item, *args, **kwargs):
        computed.append(int(item["row"].get("origin_pos")))
        return original(item, *args, **kwargs)

    monkeypatch.setattr(runner, "_fit_predict_origin", _spy)

    resumed = _run(checkpoint_path=cell, policy="direct")
    resumed_frame = _sorted_frame(resumed.to_frame())

    # Nothing recomputed; frame reconstructed entirely from the checkpoint.
    assert computed == []
    assert list(resumed_frame["origin_pos"]) == list(full_frame["origin_pos"])
    np.testing.assert_allclose(
        resumed_frame["prediction"].to_numpy(dtype=float),
        full_frame["prediction"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-12,
    )
