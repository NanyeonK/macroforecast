from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, pipeline_spec, run_pipeline


class _MeanFit:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.value, dtype=float)


def _fast_mean_fit(X: pd.DataFrame, y: pd.Series, *, offset: float = 0.0) -> _MeanFit:
    return _MeanFit(float(np.nanmean(np.asarray(y, dtype=float))) + float(offset))


def _flag_slow_mean_fit(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    flag_path: str,
    sleep_seconds: float = 5.0,
    offset: float = 0.0,
) -> _MeanFit:
    if Path(flag_path).exists():
        time.sleep(float(sleep_seconds))
    return _fast_mean_fit(X, y, offset=offset)


def _exit_worker_fit(X: pd.DataFrame, y: pd.Series) -> _MeanFit:
    os._exit(88)


def _bundle(n: int = 72) -> object:
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    x = np.linspace(0.0, 1.0, n)
    frame = pd.DataFrame(
        {
            "y": 1.0 + 0.5 * x,
            "x1": x,
        },
        index=idx,
    )
    return mf.data.custom_dataset(frame, transform_codes={"y": 1, "x1": 1})


def _window() -> Any:
    return mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=30),
        val=mf.window.val_last_block(size=8),
        test=mf.window.test_origins(horizon=1, step=12),
    )


def _features() -> Any:
    return mf.feature_engineering.feature_spec(
        target="y",
        predictors=["x1"],
        lags=1,
        target_lags=(0, 1),
    )


def _fast_model(name: str = "fast_mean") -> Any:
    return mf.models.custom_model(
        name,
        _fast_mean_fit,
        default_params={"offset": 0.0},
        mf_digest=f"{name}-v1",
    )


def _slow_model(flag_path: Path) -> Any:
    return mf.models.custom_model(
        "flag_slow_mean",
        _flag_slow_mean_fit,
        default_params={
            "flag_path": str(flag_path),
            "sleep_seconds": 5.0,
            "offset": 0.25,
        },
        mf_digest="flag-slow-mean-v1",
    )


def _exit_model() -> Any:
    return mf.models.custom_model(
        "exit_worker",
        _exit_worker_fit,
        mf_digest="exit-worker-v1",
    )


def _spec(
    *,
    arms: list[Arm] | None = None,
    horizons: tuple[int, ...] = (1, 2, 3, 4),
    n_jobs: int = 1,
    parallel_cell_timeout: float | int | None = 3600.0,
    result_store: Path | None = None,
) -> Any:
    return pipeline_spec(
        data=_bundle(),
        targets=["y"],
        horizons=horizons,
        window=_window(),
        arms=arms
        if arms is not None
        else [
            Arm("FAST_A", model=_fast_model("fast_mean_a"), features=_features(), is_benchmark=True),
            Arm("FAST_B", model=_fast_model("fast_mean_b"), features=_features()),
        ],
        evaluation=EvalSpec(benchmark="FAST_A", metrics=("rmse",), tests=()),
        save_models=False,
        result_store=result_store,
        n_jobs=n_jobs,
        parallel_cell_timeout=parallel_cell_timeout,
        provenance_level="basic",
    )


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        col
        for col in ("target", "contender", "horizon", "origin", "date")
        if col in frame.columns
    ]
    return (
        frame.sort_values(keys, kind="mergesort")
        .reset_index(drop=True)
        .reindex(sorted(frame.columns), axis=1)
    )


@pytest.mark.timeout(20)
def test_parallel_cell_timeout_is_bounded_and_loud(tmp_path: Path) -> None:
    flag = tmp_path / "slow.flag"
    flag.write_text("slow\n")
    arms = [
        Arm("FAST_A", model=_fast_model("timeout_fast"), features=_features(), is_benchmark=True),
        Arm("SLOW", model=_slow_model(flag), features=_features()),
    ]

    started = time.monotonic()
    with pytest.warns(RuntimeWarning):
        report = run_pipeline(
            _spec(
                arms=arms,
                n_jobs=2,
                parallel_cell_timeout=0.5,
                result_store=tmp_path / "results",
            )
        )

    assert time.monotonic() - started < 15.0
    assert not report.forecasts.empty
    errors = [str(record["error"]) for record in report.failed_cells]
    assert any("ParallelExecutorTimeout" in error for error in errors)


@pytest.mark.timeout(20)
def test_broken_parallel_worker_is_bounded_and_loud() -> None:
    arms = [
        Arm("FAST_A", model=_fast_model("broken_fast"), features=_features(), is_benchmark=True),
        Arm("EXIT", model=_exit_model(), features=_features()),
    ]

    with pytest.warns(RuntimeWarning):
        report = run_pipeline(
            _spec(arms=arms, horizons=(1, 2), n_jobs=2, parallel_cell_timeout=5.0)
        )

    errors = [str(record["error"]) for record in report.failed_cells]
    assert errors
    assert any(
        "BrokenProcessPool" in error or "ParallelExecutorError" in error
        for error in errors
    )


@pytest.mark.timeout(25)
def test_result_store_resume_after_parallel_timeout(tmp_path: Path) -> None:
    flag = tmp_path / "slow.flag"
    flag.write_text("slow\n")
    store = tmp_path / "results"
    arms = [
        Arm("FAST_A", model=_fast_model("resume_fast"), features=_features(), is_benchmark=True),
        Arm("SLOW", model=_slow_model(flag), features=_features()),
    ]

    with pytest.warns(RuntimeWarning):
        first = run_pipeline(
            _spec(
                arms=arms,
                n_jobs=2,
                parallel_cell_timeout=0.5,
                result_store=store,
            )
        )

    assert first.failed_cells
    assert first.provenance["result_store"]["n_computed"] > 0

    flag.unlink()
    second = run_pipeline(
        _spec(
            arms=arms,
            n_jobs=2,
            parallel_cell_timeout=10.0,
            result_store=store,
        )
    )

    assert not second.failed_cells
    assert second.provenance["result_store"]["n_reused"] > 0
    assert second.provenance["result_store"]["n_computed"] > 0
    assert not second.forecasts.empty


def test_parallel_numbers_match_serial_with_timeout() -> None:
    serial = run_pipeline(_spec(n_jobs=1))
    parallel = run_pipeline(_spec(n_jobs=2, parallel_cell_timeout=10.0))

    pd.testing.assert_frame_equal(_sorted(serial.forecasts), _sorted(parallel.forecasts))
    pd.testing.assert_frame_equal(_sorted(serial.accuracy), _sorted(parallel.accuracy))


def test_parallel_cell_timeout_validation() -> None:
    assert _spec().parallel_cell_timeout == 3600.0
    assert _spec(parallel_cell_timeout=2).parallel_cell_timeout == 2.0
    assert _spec(parallel_cell_timeout=0.25).parallel_cell_timeout == 0.25
    assert _spec(parallel_cell_timeout=None).parallel_cell_timeout is None

    for bad in (0, -1, True, False, math.inf, -math.inf, math.nan, "1"):
        with pytest.raises(ValueError, match="parallel_cell_timeout"):
            _spec(parallel_cell_timeout=bad)  # type: ignore[arg-type]
