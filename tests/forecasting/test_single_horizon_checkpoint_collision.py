"""Regression tests for the single-horizon checkpoint-collision bug.

A single-horizon ``run(..., horizons=[h])`` must forecast horizon ``h``, not the
default horizon 1. The original defect was that the single-horizon execution path
wrote its checkpoint origins directly under ``checkpoint_path`` (no ``h<h>``
namespace), while the multi-horizon path namespaced each horizon under
``checkpoint_path/h<h>``. Origin positions are horizon-independent, so a per-cell
checkpoint directory reused across distinct single-horizon runs (the per-horizon
loop used by the ML-Useful replication runner) collided: the first horizon's lean
records (carrying horizon=1 and an origin->date gap of 1) were loaded as
"already done" for every later horizon, silently forecasting horizon 1.

These tests assert that, with a shared per-cell checkpoint directory, each
single-horizon run yields an origin->date gap and a ``horizon`` column equal to
the requested horizon, both through the pipeline and through the direct runner.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, pipeline_spec, run_arms


def _bundle(n: int = 160):
    idx = pd.date_range("1985-01-31", periods=n, freq="ME", name="date")
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, n)
    frame = pd.DataFrame(
        {"y": 1.0 + 2.0 * x + rng.standard_normal(n) * 0.05, "x1": x},
        index=idx,
    )
    return mf.data.custom_dataset(frame, transform_codes={"y": 1, "x1": 1})


def _window(horizon: int):
    return mf.window.from_cutoffs(
        estimation_start="1985-01", test_start="1990-01", test_end="1996-12",
        mode="expanding", horizon=horizon, embargo=0, retrain_every=12,
        val_method="last_block", val_size=12,
    )


def _features():
    return mf.feature_engineering.feature_spec(
        target="y", predictors=["x1"], lags=1, target_lags=(0, 1)
    )


def _month_gap(origin, date) -> int:
    o = pd.Timestamp(origin)
    d = pd.Timestamp(date)
    return (d.year - o.year) * 12 + (d.month - o.month)


def _gaps(frame: pd.DataFrame) -> set[int]:
    frame = frame.dropna(subset=["prediction"])
    return {_month_gap(o, d) for o, d in zip(frame["origin"], frame["date"])}


@pytest.mark.parametrize("horizon", [1, 3, 12])
def test_pipeline_single_horizon_uses_requested_horizon(horizon: int) -> None:
    spec = pipeline_spec(
        data=_bundle(), targets=["y"], horizons=[horizon], window=_window(horizon),
        arms=[Arm("AR", model="ar", features=_features())],
        evaluation=EvalSpec(benchmark="AR"), save_models=False,
    )
    master = run_arms(spec).dropna(subset=["prediction"])
    assert not master.empty
    assert set(master["horizon"]) == {horizon}
    assert _gaps(master) == {horizon}


@pytest.mark.parametrize("horizon", [1, 3, 12])
def test_pipeline_single_horizon_shared_checkpoint_dir(
    tmp_path: Path, horizon: int
) -> None:
    """A shared per-cell checkpoint dir must not leak one horizon into another.

    This mirrors the ML-Useful replication loop that called run_pipeline once per
    horizon with the SAME checkpoint_dir; the bug reused horizon 1's origins for
    every later horizon.
    """
    ckpt = tmp_path / "shared_ckpt"
    # Run several horizons in sequence, all sharing the one checkpoint directory.
    for h in (1, 3, 12):
        spec = pipeline_spec(
            data=_bundle(), targets=["y"], horizons=[h], window=_window(h),
            arms=[Arm("AR", model="ar", features=_features())],
            evaluation=EvalSpec(benchmark="AR"), save_models=False,
            checkpoint_dir=str(ckpt),
        )
        master = run_arms(spec).dropna(subset=["prediction"])
        if h == horizon:
            assert not master.empty
            assert set(master["horizon"]) == {horizon}
            assert _gaps(master) == {horizon}


@pytest.mark.parametrize("horizon", [1, 3, 12])
def test_direct_run_single_element_horizons_list(
    tmp_path: Path, horizon: int
) -> None:
    """Direct mf.forecasting.run with a single-element horizons list (and a shared
    checkpoint dir) yields gap == horizon and horizon column == horizon."""
    bundle = _bundle()
    panel = bundle.panel
    ckpt = tmp_path / "cell"
    for h in (1, 3, 12):
        result = mf.forecasting.run(
            panel,
            "ar",
            window=_window(h),
            features=_features(),
            target="y",
            horizons=[h],
            save_models=False,
            checkpoint_path=ckpt,
        )
        if h == horizon:
            frame = result.to_frame().dropna(subset=["prediction"])
            assert not frame.empty
            assert set(frame["horizon"]) == {horizon}
            assert _gaps(frame) == {horizon}


def test_multi_horizon_stays_correct(tmp_path: Path) -> None:
    """The known-correct multi-horizon path keeps each horizon's gap == horizon."""
    spec = pipeline_spec(
        data=_bundle(), targets=["y"], horizons=[1, 3, 12], window=_window(1),
        arms=[Arm("AR", model="ar", features=_features())],
        evaluation=EvalSpec(benchmark="AR"), save_models=False,
        checkpoint_dir=str(tmp_path / "multi"),
    )
    master = run_arms(spec).dropna(subset=["prediction"])
    assert set(master["horizon"]) == {1, 3, 12}
    for h in (1, 3, 12):
        sub = master[master["horizon"] == h]
        assert _gaps(sub) == {h}
