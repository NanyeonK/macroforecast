"""AUTO CPU-parallelism allocator: core/work split and the n_jobs="auto" flow.

``auto_parallelism`` splits a core budget into (cell_workers, model_threads) whose
product never exceeds the budget, favouring cell-level parallelism first. The
pipeline spec resolves ``n_jobs="auto"`` into a concrete cell-worker count plus a
per-cell model-internal thread budget, and the resulting run is numerically
identical to the serial / explicit-parallel paths (the auto split changes only the
thread COUNT, not the result). Windows are kept tiny so this runs in well under a
minute and does not starve the long-running replication processes.
"""
import os

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, auto_parallelism, pipeline_spec, run_pipeline


# --------------------------------------------------------------------------- #
# auto_parallelism: the core/work split arithmetic
# --------------------------------------------------------------------------- #

def test_auto_parallelism_splits_cores():
    # Cell-level parallelism first; leftover cores become model-internal threads.
    assert auto_parallelism(20, cores=48) == (20, 2)   # 20 workers x 2 threads = 40 <= 48
    assert auto_parallelism(100, cores=48) == (48, 1)  # more cells than cores -> all workers
    assert auto_parallelism(5, cores=48) == (5, 9)     # few cells -> deep per-cell threads
    assert auto_parallelism(60, cores=48) == (48, 1)   # cells slightly > cores -> saturate
    assert auto_parallelism(1, cores=8) == (1, 8)      # one cell -> all cores to the model


def test_auto_parallelism_never_oversubscribes():
    # The product of the two factors must never exceed the core budget.
    for n_cells in (1, 3, 5, 7, 20, 48, 60, 100):
        for cores in (1, 4, 8, 48, 64):
            workers, threads = auto_parallelism(n_cells, cores=cores)
            assert workers >= 1 and threads >= 1
            assert workers * threads <= cores


def test_auto_parallelism_reserve_respected():
    # reserve holds back cores before the split.
    assert auto_parallelism(100, cores=48, reserve=8) == (40, 1)
    assert auto_parallelism(5, cores=48, reserve=8) == (5, 8)   # cores=40, 40//5=8
    assert auto_parallelism(20, cores=48, reserve=8) == (20, 2)  # cores=40, 40//20=2
    # reserve cannot drive the budget below one core.
    assert auto_parallelism(4, cores=4, reserve=10) == (1, 1)


def test_auto_parallelism_cores_default_is_affinity():
    affinity = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    workers, threads = auto_parallelism(1)  # one cell -> all cores to model threads
    assert workers == 1
    assert threads == affinity


def test_auto_parallelism_default_falls_back_without_sched_getaffinity(monkeypatch):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 6)

    assert auto_parallelism(2) == (2, 3)


# --------------------------------------------------------------------------- #
# pipeline_spec(n_jobs="auto") resolution + invalid n_jobs
# --------------------------------------------------------------------------- #

def _bundle(n: int = 84) -> object:
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, n)
    frame = pd.DataFrame(
        {
            "y1": 1.0 + 2.0 * x + rng.standard_normal(n) * 0.05,
            "x1": x,
            "x2": np.cos(np.linspace(0.0, 6.0, n)),
        },
        index=idx,
    )
    return mf.data.custom_dataset(frame, transform_codes={"y1": 1, "x1": 1, "x2": 1})


def _spec(n_jobs):
    feats = mf.feature_engineering.feature_spec(
        target="y1", predictors=["x1", "x2"], lags=1, target_lags=(0, 1)
    )
    w = mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=36),
        val=mf.window.val_last_block(size=12),
        test=mf.window.test_origins(horizon=1, step=6),
    )
    return pipeline_spec(
        data=_bundle(),
        targets=["y1"],
        horizons=[1, 3],
        window=w,
        arms=[
            Arm("AR", model="ar", features=feats),
            Arm("OLS", model="ols", features=feats),
            Arm("RIDGE", model="ridge", features=feats),
        ],
        evaluation=EvalSpec(benchmark="AR"),
        save_models=False,
        n_jobs=n_jobs,
    )


def test_pipeline_spec_auto_resolves_to_int_and_sets_model_threads():
    spec = _spec(n_jobs="auto")
    # n_jobs resolves to a concrete positive int (cell workers).
    assert isinstance(spec.n_jobs, int)
    assert spec.n_jobs >= 1
    # model_threads is a positive int set by the allocator.
    assert isinstance(spec.model_threads, int)
    assert spec.model_threads >= 1
    # Never oversubscribes the affinity budget.
    core_budget = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    assert spec.n_jobs * spec.model_threads <= core_budget
    # 1 target x 3 arms x 2 horizons = 6 cells -> at most 6 workers.
    assert spec.n_jobs <= 6

    # An explicit int leaves model_threads at the default 1.
    assert _spec(n_jobs=1).model_threads == 1
    assert _spec(n_jobs=4).model_threads == 1


@pytest.mark.parametrize("bad", [0, -1, "bogus", 1.5, True])
def test_pipeline_spec_invalid_n_jobs_raises(bad):
    with pytest.raises(ValueError):
        _spec(n_jobs=bad)


# --------------------------------------------------------------------------- #
# determinism: n_jobs="auto" == n_jobs=1 == n_jobs=4 numerically
# --------------------------------------------------------------------------- #

_SORT_KEYS = ["target", "contender", "horizon", "origin"]


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [k for k in _SORT_KEYS if k in frame.columns]
    return (
        frame.sort_values(keys, kind="mergesort")
        .reset_index(drop=True)
        .reindex(sorted(frame.columns), axis=1)
    )


def test_auto_forecasts_identical_to_serial_and_parallel():
    serial = run_pipeline(_spec(n_jobs=1))
    parallel = run_pipeline(_spec(n_jobs=4))
    auto = run_pipeline(_spec(n_jobs="auto"))

    fs = _sorted(serial.forecasts)
    fp = _sorted(parallel.forecasts)
    fa = _sorted(auto.forecasts)

    pd.testing.assert_frame_equal(fs, fa, check_like=True)
    pd.testing.assert_frame_equal(fp, fa, check_like=True)
