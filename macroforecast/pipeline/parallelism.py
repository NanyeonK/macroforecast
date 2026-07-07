"""AUTO CPU-parallelism allocator for the pipeline.

The pipeline fans ``(arm x target x horizon)`` *cells* across a process pool and,
inside each worker, the parallelizable models (tree ensembles) can spawn their own
internal threads. Naively combining the two oversubscribes the CPU: ``N`` cell
workers each spawning ``cpu_count`` model threads gives ``N x cpu_count`` threads.

``auto_parallelism`` splits the available cores into two factors -- cell workers and
per-cell model-internal threads -- whose product never exceeds the core budget. It
favours cell-level parallelism first (one worker per cell up to the core count) and
hands the leftover cores to each worker as model-internal threads.
"""
from __future__ import annotations

import os


def auto_parallelism(
    n_cells: int, *, cores: int | None = None, reserve: int = 0
) -> tuple[int, int]:
    """Return ``(cell_workers, model_threads)`` saturating ``cores``.

    Cell-level parallelism comes first: one worker per cell up to the core budget.
    Whatever cores remain become per-cell model-internal threads for the
    parallelizable models (random_forest, gradient_boosting, xgboost, lightgbm).
    The product ``cell_workers * model_threads`` is always ``<= cores``, so the CPU
    is never oversubscribed.

    Parameters
    ----------
    n_cells:
        Number of independent ``(arm x target x horizon)`` cells to schedule.
    cores:
        Core budget. Defaults to the affinity count
        (``len(os.sched_getaffinity(0))``) -- the cores this process may actually
        run on, which respects cgroup/taskset pinning.
    reserve:
        Cores to hold back (e.g. for the parent process / other work) before the
        split. ``cores`` is reduced by ``reserve`` (floored at 1).
    """
    if cores is None:
        # Affinity count, not os.cpu_count(): respects cgroup / taskset pinning so
        # we saturate only the cores this process is actually allowed to use. On
        # platforms without sched_getaffinity (macOS/Windows), fall back to the
        # visible CPU count.
        get_affinity = getattr(os, "sched_getaffinity", None)
        cores = len(get_affinity(0)) if get_affinity is not None else (os.cpu_count() or 1)
    cores = max(1, cores - reserve)
    # One worker per cell, capped at the core budget; at least one worker.
    workers = max(1, min(int(n_cells), cores))
    # Leftover cores per worker become model-internal threads; at least one.
    model_threads = max(1, cores // workers)
    return workers, model_threads
