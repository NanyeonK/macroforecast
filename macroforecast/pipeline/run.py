"""Pipeline execution: schedule and run cells into the master forecast frame (Stage 1+).

The pipeline MANAGES execution by enumerating cells -- one ``arm`` applied to one
``target`` over the window for a horizon-group -- and executing each cell with one
atomic ``run()`` call. The serial backend groups all horizons of a (target, arm)
into one cell; the parallel backend splits one horizon per cell.
"""
from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from macroforecast.pipeline.spec import (
    Arm,
    PipelineSpec,
    ResolvedTarget,
    _model_default_name,
)


def _safe_segment(value: str) -> str:
    """Make a target/arm name safe to use as a single filesystem path segment."""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("_") or "x"


def _cell_checkpoint_path(spec: PipelineSpec, arm: Arm, target: ResolvedTarget):
    """Per-(target, arm) checkpoint directory, or None when disabled.

    ``run()`` appends an ``h<h>`` subdirectory for each horizon when the spec
    carries more than one horizon, so the final layout is
    ``<checkpoint_dir>/<target>__<arm>/h<h>/origin_<pos>.parquet``.
    """
    if spec.checkpoint_dir is None:
        return None
    cell = f"{_safe_segment(target.name)}__{_safe_segment(arm.name)}"
    return Path(spec.checkpoint_dir) / cell


def _run_one_arm_target(
    spec: PipelineSpec,
    arm: Arm,
    target: ResolvedTarget,
    preprocessing_cache=None,
    preprocessing_store=None,
    horizons: "Sequence[int] | None" = None,
) -> pd.DataFrame:
    """Execute one cell: a single arm applied to a single resolved target.

    By default every horizon in ``spec.horizons`` is computed in one consolidated
    multi-horizon ``run()`` call (the serial cell, sharing one
    ``preprocessing_cache`` across horizons). When ``horizons`` is given, only
    those horizons are computed -- the parallel path passes a single horizon per
    cell so independent processes each execute just their cell.
    """
    import dataclasses as _dc

    from macroforecast.forecasting import run

    run_horizons = list(spec.horizons if horizons is None else horizons)

    # A multi-target pipeline runs each arm for every target, but an arm's feature
    # spec carries a single target; align it (and its transform) to this target.
    features = arm.features
    if features is not None:
        needs_retarget = (
            getattr(features, "target", None) != target.name
            or bool(getattr(features, "targets", ()))
        )
        if needs_retarget:
            kwargs: dict[str, Any] = {"target": target.name}
            if getattr(features, "targets", None):
                kwargs["targets"] = ()
            try:
                features = _dc.replace(features, **kwargs)
            except Exception as exc:  # surface a real misconfiguration
                raise ValueError(
                    f"could not re-target feature spec of arm {arm.name!r} "
                    f"to {target.name!r}: {exc}"
                ) from exc

    result = run(
        spec.data,
        arm.model,
        window=arm.window if getattr(arm, "window", None) is not None else spec.window,
        preprocessing=arm.preprocessing if arm.preprocessing is not None else spec.preprocessing,
        preprocessing_policy=(arm.preprocessing_policy if arm.preprocessing is not None
                              else spec.preprocessing_policy),
        features=features,
        feature_policy=arm.feature_policy,
        params=arm.params,
        model_selection=arm.model_selection,
        model_selection_metric=arm.model_selection_metric,
        target=target.name,
        horizons=run_horizons,
        forecast_policy=target.policy,
        target_transform=target.transform,
        save_models=spec.save_models,
        model_store=spec.model_store,
        preprocessing_cache=preprocessing_cache,
        preprocessing_store=preprocessing_store,
        checkpoint_path=_cell_checkpoint_path(spec, arm, target),
    )
    frame = result.to_frame().copy()
    frame["arm"] = arm.name
    # A contender IS exactly an arm (one model per arm), so the contender label is
    # always the arm name regardless of the underlying model row.
    frame["contender"] = arm.name
    # ensure the target column carries the resolved target name
    if "target" not in frame.columns:
        frame["target"] = target.name
    return frame


def _empty_arm_warning(arm_name: str, target_name: str) -> str:
    """The diagnostic emitted when an arm yields zero forecast rows for a target."""
    return (
        f"pipeline arm {arm_name!r} produced ZERO forecast rows for "
        f"target {target_name!r}; it will be absent from the evaluation. "
        "Check the arm's feature spec / preprocessing (an all-NaN feature "
        "block over every origin empties the fit sample)."
    )


# --------------------------------------------------------------------------- #
# Unified cell run-manager
# --------------------------------------------------------------------------- #
# The pipeline MANAGES atomic ``run()`` calls. The managed unit is a "cell":
# ``(target, arm, horizon-group)``. Both backends enumerate the SAME cells and
# execute them through one structure; they differ only in how horizons are grouped
# and whether a preprocessing cache is shared.
#
#   * serial (n_jobs == 1): one cell per ``(target, arm)`` covering ALL horizons in
#     a single multi-horizon ``run()`` so the per-origin preprocessing cache / EM
#     state is shared across horizons (the byte-for-byte prior behavior).
#   * parallel (n_jobs > 1): one cell per ``(target, arm, horizon)`` -- a single
#     horizon per ``run()`` -- so independent processes each compute just their
#     cell. The in-memory cache is not shared across processes; by default each
#     recomputes its own EM. When ``spec.preprocessing_cache_dir`` is set, workers
#     additionally share each per-(spec, target, origin) fit through an on-disk
#     ``PreprocessorStore`` at that directory, so the EM is computed once overall.
#     Either way the forecasts are numerically identical to the serial path.
#
# Per cell: respect the per-(target, arm, horizon) checkpoint dirs; if a cell
# raises, record the failure (cell identity + error) and CONTINUE the rest of the
# set rather than aborting. Failures are surfaced on ``PipelineReport.failed_cells``
# (and mirrored into ``leakage_audit``).


class _Cell(NamedTuple):
    """One cell, the managed execution unit: a (target, arm, horizon-group)."""

    target_idx: int
    arm_idx: int
    horizons: tuple[int, ...]


def _enumerate_cells(spec: PipelineSpec) -> list[_Cell]:
    """Enumerate cells in the deterministic (target -> arm -> horizon) visit order.

    Serial groups all horizons per (target, arm); parallel splits one horizon per
    cell. The reassembly order is the enumeration order, so the master frame's row
    order is identical regardless of backend or worker completion order.
    """
    horizons = tuple(int(h) for h in spec.horizons)
    cells: list[_Cell] = []
    for t_idx, _target in enumerate(spec.targets):
        for a_idx, _arm in enumerate(spec.arms):
            if spec.n_jobs > 1:
                cells.extend(_Cell(t_idx, a_idx, (h,)) for h in horizons)
            else:
                cells.append(_Cell(t_idx, a_idx, horizons))
    return cells


# Relative per-cell cost weights by model family, used only to ORDER dispatch
# (longest-processing-time-first). Exact values do not affect results, only the
# order in which independent cells are submitted to the pool.
_FAMILY_COST_WEIGHT = {
    "tree": 8.0,
    "neural": 8.0,
    "support_vector": 6.0,
    "composite": 5.0,
    "nonparametric": 4.0,
    "mixed_frequency": 3.0,
    "assemblage": 2.5,
    "factor": 2.0,
    "volatility": 2.0,
    "spline": 2.0,
    "linear": 1.5,
    "timeseries": 1.0,
}
_DEFAULT_COST_WEIGHT = 2.0
_MODEL_FAMILY_CACHE: dict[str, str] = {}


def _model_family(name: str | None) -> str | None:
    """Look up a model's family from the registry, cached. Best-effort."""
    if not name:
        return None
    if not _MODEL_FAMILY_CACHE:
        try:
            from macroforecast import list_model_specs

            df = list_model_specs()
            for record_name, family in zip(df["name"], df["family"]):
                _MODEL_FAMILY_CACHE[str(record_name)] = str(family)
        except Exception:
            # Leave the cache empty; callers fall back to the default weight.
            _MODEL_FAMILY_CACHE["__unavailable__"] = ""
    return _MODEL_FAMILY_CACHE.get(name)


def _cell_cost(spec: PipelineSpec, cell: _Cell) -> float:
    """A cheap proxy for how long a cell takes, for dispatch ordering only.

    Within one ``run_pipeline`` call every cell shares the forecast policy, so the
    cost ordering is driven by model family (tree ensembles and nets dominate) and
    the longest horizon (more iterated steps and a larger training tail). The proxy
    never affects numerical results -- it only decides submission order so the
    heaviest cells start first and are not stranded alone at the tail.
    """
    arm = spec.arms[cell.arm_idx]
    try:
        name = _model_default_name(arm.model)
    except Exception:
        name = None
    family = _model_family(name)
    weight = (
        _FAMILY_COST_WEIGHT.get(family, _DEFAULT_COST_WEIGHT)
        if family is not None
        else _DEFAULT_COST_WEIGHT
    )
    horizon = max(cell.horizons) if cell.horizons else 1
    return weight * float(horizon)


def _lpt_dispatch_order(spec: PipelineSpec, cells: list[_Cell]) -> list[_Cell]:
    """Order cells by descending cost (longest-processing-time-first).

    Ties keep the canonical enumeration order (stable sort), so the schedule is
    deterministic. Reassembly still iterates the canonical ``cells`` list, so this
    ordering changes only the wall-clock makespan, never the output.
    """
    return sorted(cells, key=lambda c: _cell_cost(spec, c), reverse=True)


def _effective_preprocessing_policy(spec: PipelineSpec, arm: Arm):
    """The ``StagePolicy`` that will actually govern this arm's preprocessing fit.

    Mirrors the exact override rule ``_run_one_arm_target`` uses when it calls
    ``run()``: an arm with its own ``preprocessing`` override also supplies its own
    ``preprocessing_policy`` (the spec-level policy never applies to an override
    arm); otherwise the arm inherits the spec-level policy. The result is resolved
    (string/``None``/``StagePolicy`` -> a concrete ``StagePolicy``) through the same
    ``resolve_stage_policy`` + ``default_preprocessing_scope`` that ``run()`` itself
    uses, so this is not a re-derivation that could drift from the runner -- it is
    the identical resolution, called one layer earlier.
    """
    from macroforecast.meta import get_config
    from macroforecast.window.policy import resolve_stage_policy

    effective_preprocessing = arm.preprocessing if arm.preprocessing is not None else spec.preprocessing
    if effective_preprocessing is None:
        return None
    effective_policy = (
        arm.preprocessing_policy if arm.preprocessing is not None else spec.preprocessing_policy
    )
    default_scope = str(get_config()["default_preprocessing_scope"])
    return resolve_stage_policy(effective_policy, default_scope=default_scope)


def _execute_cell(
    spec: PipelineSpec, cell: _Cell, *, preprocessing_cache=None
) -> pd.DataFrame:
    """Run ONE cell as a single (multi- or single-horizon) ``run()`` call.

    When ``spec.preprocessing_cache_dir`` is set, construct a shared on-disk
    ``PreprocessorStore`` rooted there and thread it into ``run()``. Both backends
    call this function, so each parallel worker (which receives the pickled ``spec``,
    carrying ``preprocessing_cache_dir``) independently constructs a store pointing
    at the SAME directory and shares the persisted per-origin fits via the
    filesystem. The store is a thin, content-addressed path wrapper, so there is no
    need to pickle the store object itself across processes.

    The store is namespaced by this arm's EFFECTIVE ``preprocessing_policy``
    (``StagePolicy.to_dict()``, not just its ``scope`` string, so a
    ``fixed_reference``/``custom`` policy that differs only in its reference
    bounds or selector is also isolated) -- see ``_effective_preprocessing_policy``.
    Without this, the store key hashes only ``(PreprocessSpec, target,
    origin_pos)``; two runs sharing one ``preprocessing_cache_dir`` but using
    different scopes (e.g. one ``origin_available``, one ``fit_window``) for the
    same spec would otherwise silently serve one run's fit to the other.
    """
    arm = spec.arms[cell.arm_idx]
    target = spec.targets[cell.target_idx]
    store = None
    if spec.preprocessing_cache_dir:
        from macroforecast.preprocessing.cache import PreprocessorStore

        policy = _effective_preprocessing_policy(spec, arm)
        namespace = policy.to_dict() if policy is not None else None
        # Passed to every arm, including arms with a per-arm ``preprocessing``
        # override: the store key hashes the per-arm resolved spec, so an override
        # arm's distinct spec yields a distinct key and never collides with the
        # shared-spec arms. The namespace additionally isolates arms/specs that
        # resolve to a different effective preprocessing_policy.
        store = PreprocessorStore(spec.preprocessing_cache_dir, namespace=namespace)
    return _run_one_arm_target(
        spec, arm, target,
        preprocessing_cache=preprocessing_cache,
        preprocessing_store=store,
        horizons=list(cell.horizons),
    )


def _parallel_cell_worker(
    args: "tuple[PipelineSpec, _Cell]",
) -> "tuple[_Cell, pd.DataFrame | None, str | None]":
    """Module-level worker: execute ONE cell (its single ``run()``) in a subprocess,
    returning any error text.

    Caps nested BLAS/OpenMP threads to one so a pool of ``n_jobs`` processes does
    not oversubscribe the cores. Returns ``(cell, frame, error)`` where exactly one
    of ``frame``/``error`` is set, so the parent isolates per-cell failures.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")
    spec, cell = args
    # Pin this worker's model-internal thread budget from the AUTO allocator so the
    # parallelizable models (RF/GBM/XGB/LGBM) inside the cell use exactly
    # spec.model_threads threads -- using the leftover cores while keeping
    # cell_workers * model_threads <= cores (no oversubscription). Only changes the
    # thread COUNT, never the numerical result (tree training is deterministic in
    # random_state regardless of n_jobs). This runs in the worker's own process, so
    # it does not touch the parent's meta config.
    import macroforecast as mf

    mf.meta.configure(n_jobs=int(spec.model_threads))
    try:
        # No in-memory cache across processes (preprocessing_cache=None); each cell
        # recomputes its own preprocessing unless spec.preprocessing_cache_dir is set,
        # in which case _execute_cell builds an on-disk PreprocessorStore the workers
        # share so each per-(spec, target, origin) fit is computed once overall.
        return cell, _execute_cell(spec, cell, preprocessing_cache=None), None
    except Exception as exc:  # isolate the failure; the parent records it
        return cell, None, f"{type(exc).__name__}: {exc}"


def _cell_failure(spec: PipelineSpec, cell: _Cell, error: str) -> dict[str, Any]:
    """A structured record of one failed cell (identity + error)."""
    arm = spec.arms[cell.arm_idx]
    target = spec.targets[cell.target_idx]
    return {
        "target": target.name,
        "arm": arm.name,
        "horizons": list(cell.horizons),
        "error": error,
    }


def _empty_cell_warning(target_name: str, horizon: int, arms: "Sequence[str]") -> str:
    """The diagnostic emitted when a (target, horizon) cell produces zero rows."""
    arms_txt = ", ".join(repr(a) for a in arms)
    return (
        f"pipeline (target={target_name!r}, horizon={int(horizon)}) produced ZERO "
        f"forecast rows for arm(s) {arms_txt}; this (target, horizon) will be "
        "SILENTLY ABSENT from the evaluation. The cell ran without error, so this "
        "is a zero-scorable-origin / data-availability condition rather than a "
        "failure -- inspect target availability at this horizon over the window."
    )


def _find_empty_cells(
    spec: PipelineSpec,
    master: pd.DataFrame,
    failed_keys: "set[tuple[str, str]]",
) -> list[dict[str, Any]]:
    """Find (target, horizon) cells that ran but yielded zero forecast rows.

    Compares the expected (target, horizon) grid against the (target, horizon)
    pairs actually present in the assembled master. A pair is reported as empty
    only when at least one arm was expected to populate it AND that arm did not
    FAIL (failures are surfaced separately on ``failed_cells``). Each record lists
    the arm(s) that produced no rows for that (target, horizon).
    """
    horizons = [int(h) for h in spec.horizons]
    # (target, horizon) -> contenders present in the master
    present: dict[tuple[str, int], set[str]] = {}
    if not master.empty and {"target", "horizon", "contender"}.issubset(master.columns):
        for (tgt, hor), grp in master.groupby(["target", "horizon"], dropna=False):
            present[(str(tgt), int(hor))] = set(grp["contender"].unique())

    empty: list[dict[str, Any]] = []
    for target in spec.targets:
        for h in horizons:
            here = present.get((target.name, h), set())
            missing_arms = [
                arm.name
                for arm in spec.arms
                if arm.name not in here
                and (arm.name, target.name) not in failed_keys
            ]
            if missing_arms:
                empty.append(
                    {"target": target.name, "horizon": h, "arms": missing_arms}
                )
    return empty


def _resolve_run_cache_dir(
    spec: PipelineSpec,
) -> "tuple[PipelineSpec, str | None]":
    """Resolve the effective ``preprocessing_cache_dir`` for one execution of *spec*.

    ``preprocessing_cache_dir`` has three states:

    - an explicit ``str``: used as-is (unchanged prior behavior).
    - ``False``: explicit opt-out. Never auto-create a store, matching the
      pre-existing parallel behavior where each worker recomputes its own EM
      with no shared cache. ``spec`` is returned unchanged (``_execute_cell``
      already treats a falsy ``preprocessing_cache_dir`` as "no store").
    - ``None`` (the ``pipeline_spec`` default, meaning "not explicitly
      configured"): when ``spec.n_jobs > 1`` this auto-creates a run-scoped
      temporary directory so parallel workers still share the per-origin EM
      dedup that is otherwise silently lost (each worker previously received
      ``preprocessing_cache=None`` with no fallback). When ``n_jobs == 1`` this
      is a no-op -- the serial backend already shares fits via its own
      in-memory ``target_caches``.

    Returns ``(run_spec, created_tmp_dir)``. The caller must remove
    ``created_tmp_dir`` (if not ``None``) once the run completes, via a
    ``finally`` block -- this function only creates it, never cleans it up.
    """
    import dataclasses as _dc
    import tempfile as _tempfile

    if spec.preprocessing_cache_dir is None and spec.n_jobs > 1:
        tmp_dir = _tempfile.mkdtemp(prefix="macroforecast_preprocessing_cache_")
        return _dc.replace(spec, preprocessing_cache_dir=tmp_dir), tmp_dir
    return spec, None


def _run_cells(
    spec: PipelineSpec,
) -> "tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]":
    """Execute every cell, isolating per-cell failures.

    Returns ``(master, failed_cells, empty_cells)``. The two backends share this
    one enumerate-then-execute structure; the master frame, ``failed_cells`` and
    the zero-row ``empty_cells`` are assembled in the deterministic cell order
    independent of execution backend.
    """
    import shutil as _shutil
    import warnings as _warnings

    cells = _enumerate_cells(spec)
    if not cells:
        return pd.DataFrame(), [], []

    # Auto-manage a run-scoped on-disk preprocessing cache dir when the caller
    # left preprocessing_cache_dir unset and n_jobs>1 (see _resolve_run_cache_dir).
    # created_cache_dir is non-None only when THIS call created it, so only then
    # is it our responsibility to remove it once the run finishes -- everything
    # below (dispatch, reassembly, empty-cell scan) runs inside the try so the
    # cache dir stays alive for the whole run and is always cleaned up after.
    spec, created_cache_dir = _resolve_run_cache_dir(spec)
    try:
        results: dict[_Cell, pd.DataFrame] = {}
        failed: list[dict[str, Any]] = []

        if spec.n_jobs > 1:
            from concurrent.futures import ProcessPoolExecutor

            max_workers = min(spec.n_jobs, len(cells))
            # Dispatch heaviest-first (LPT) so a heavy cell is never stranded alone
            # at the tail while cores sit idle. Results are keyed by cell and
            # reassembled in the canonical order below, so this changes makespan
            # only, not output.
            dispatch = _lpt_dispatch_order(spec, cells)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for cell, frame, error in executor.map(
                    _parallel_cell_worker, [(spec, c) for c in dispatch]
                ):
                    if error is not None:
                        failed.append(_cell_failure(spec, cell, error))
                    elif frame is not None:
                        results[cell] = frame
        else:
            # One shared per-target cache: arms of the same target reuse the per-origin
            # FittedPreprocessor/_PreparedStage (Gap A's original sharing) AND the
            # per-origin fitted feature builder (Gap A promotion -- see
            # forecasting/feature_stage.py), all keyed into the SAME dict, distinctly
            # namespaced per tier. Built unconditionally (not gated on
            # ``spec.preprocessing is not None``) so feature-only pipelines -- no
            # spec-level preprocessing at all -- still get cross-arm feature-fit
            # sharing; when ``spec.preprocessing is None`` the preprocessing tier is
            # simply never touched (``_prepare_origin_panel`` short-circuits before any
            # cache read/write), so this is a no-op for that stage, unchanged from
            # before.
            #
            # An arm opts OUT of ALL sharing by overriding ``preprocessing`` OR
            # ``window``: a ``preprocessing`` override is a genuinely different
            # transform (the preprocessing cache key does not encode the spec, only
            # origin_pos, so sharing across different specs would silently serve the
            # wrong fit). A ``window`` override (a real, tested configuration -- see
            # ``tests/pipeline/test_per_arm_window.py``) changes the estimation/fit
            # row bounds at a given origin_pos; the feature-fit cache key is
            # self-verifying against that (it carries the exact fit-sample position
            # bounds, see feature_stage.py), but the PREPROCESSING cache key is NOT
            # (origin_pos alone), so a window-overriding arm must not share the
            # preprocessing tier either. Gating both tiers on the same eligibility
            # check keeps one dict, one invariant.
            target_caches: dict[str, dict[Any, Any]] = {t.name: {} for t in spec.targets}
            for cell in cells:
                arm = spec.arms[cell.arm_idx]
                target = spec.targets[cell.target_idx]
                cache = target_caches.get(target.name)
                arm_cache = (
                    cache if (arm.preprocessing is None and arm.window is None) else None
                )
                try:
                    results[cell] = _execute_cell(spec, cell, preprocessing_cache=arm_cache)
                except Exception as exc:
                    # Record the failure and CONTINUE the rest of the set rather than
                    # aborting the whole pipeline on one bad cell.
                    failed.append(
                        _cell_failure(spec, cell, f"{type(exc).__name__}: {exc}")
                    )

        # Reassemble in enumeration order; warn once per (arm, target) that yielded
        # no rows (an empty arm hides a misconfiguration). A failed cell does NOT
        # also warn as empty -- the failure is already recorded on failed_cells.
        frames: list[pd.DataFrame] = []
        seen_empty: set[tuple[str, str]] = set()
        failed_keys = {(f["arm"], f["target"]) for f in failed}
        for cell in cells:
            arm = spec.arms[cell.arm_idx]
            target = spec.targets[cell.target_idx]
            frame = results.get(cell)
            if frame is not None and not frame.empty:
                frames.append(frame)
                continue
            key = (arm.name, target.name)
            if key in failed_keys or key in seen_empty:
                continue
            seen_empty.add(key)
            _warnings.warn(
                _empty_arm_warning(arm.name, target.name),
                RuntimeWarning,
                stacklevel=2,
            )

        master = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        # Surface (target, horizon) cells that RAN but produced zero rows. These are
        # a distinct, more granular signal from the per-(arm, target) empty-arm
        # warning above: a multi-horizon arm can populate some horizons and
        # silently drop a long horizon, which the (arm, target) check cannot see.
        # Warn once per cell.
        empty_cells = _find_empty_cells(spec, master, failed_keys)
        for cell_rec in empty_cells:
            _warnings.warn(
                _empty_cell_warning(cell_rec["target"], cell_rec["horizon"], cell_rec["arms"]),
                RuntimeWarning,
                stacklevel=2,
            )

        return master, failed, empty_cells
    finally:
        if created_cache_dir is not None:
            _shutil.rmtree(created_cache_dir, ignore_errors=True)


def run_arms(spec: PipelineSpec) -> pd.DataFrame:
    """Execute every cell and concatenate into one master forecast frame.

    (The name is retained for back-compat; the managed unit is a cell -- one arm
    applied to one target over the window for a horizon-group -- not an arm.)
    Columns include arm, model, contender, target, horizon, origin, date,
    prediction, actual, target_transform, forecast_policy. Each cell runs its arm
    with its own preprocessing/features/model against its target's resolved
    (forecast_policy, target_transform).

    The pipeline MANAGES atomic ``run()`` calls over (target, arm, horizon-group)
    cells. When ``spec.n_jobs > 1`` the cells run across a process pool, one horizon
    per cell; the result is numerically identical to the serial multi-horizon path.
    Per-cell failures are isolated -- see :func:`run_pipeline` /
    ``PipelineReport.failed_cells`` for how they are surfaced.
    """
    master, _failed, _empty = _run_cells(spec)
    return master


def _panel_frame(data: Any) -> Any:
    """Best-effort extraction of the panel object (usually a DataFrame) from any
    data input (``DataBundle``/``DataSpec``/``(panel, metadata)`` tuple/bare
    ``DataFrame``). Returns whatever ``.panel`` (or the bare frame) is, without
    an ``isinstance`` narrowing, so :func:`_panel_index` below keeps its exact
    prior behavior; callers that need a ``pd.DataFrame`` specifically (the data
    provenance descriptor) narrow it themselves.
    """
    panel = getattr(data, "panel", None)
    if panel is None and isinstance(data, tuple) and data:
        panel = data[0]
    if panel is None and isinstance(data, pd.DataFrame):
        panel = data
    return panel


def _panel_index(data: Any):
    """Best-effort extraction of the panel DatetimeIndex from any data input."""
    return getattr(_panel_frame(data), "index", None)


# --------------------------------------------------------------------------- #
# Default provenance: environment/git block + data identity + spec echo
# --------------------------------------------------------------------------- #
# ``PipelineReport.provenance`` historically carried only package_version/seed/
# targets/horizons/arms/benchmark/combinations -- enough to describe WHAT ran,
# but not WHERE it ran from (no git SHA, no environment) or WHAT DATA it ran on
# (no vintage, no content identity). A referee handed only the report artifact
# could not tell which macroforecast commit produced it, nor pin the data
# vintage. This section adds that, reusing ``output.collect_provenance`` (the
# existing git/env/deps probe used by the opt-in save path) rather than
# duplicating its logic.

# Empirically measured on this project's dev host (see Wave B lane B-2 report):
# a FRED-MD-sized panel (780 rows x 130 cols, ~0.1M cells) fingerprints in
# ~1ms; a 2,000 x 400 panel in ~8ms; a deliberately huge synthetic 50,000 x
# 2,000 panel (100M cells) took ~1.3s. FRED-MD/QD/SD panels macroforecast
# actually loads are all far below this cap's cell count, so the default path
# is always the full-content digest; the cap below is a safety valve for a
# pathologically large custom panel, not something real usage is expected to
# hit.
_FINGERPRINT_FULL_CELL_CAP = 20_000_000


def _package_source_root() -> Path:
    """Directory of the RUNNING macroforecast package, for the git provenance
    probe.

    This is inside the ``macroforecast`` package tree whether it was installed
    as an editable/source checkout or from a wheel. ``collect_provenance``'s
    git probe walks up from here to find an enclosing ``.git``: when running
    from a source checkout it resolves to that checkout's commit/branch/dirty
    state; when installed from a wheel (no ``.git`` anywhere above
    site-packages) the git subprocess calls fail and it returns ``None``s --
    exactly the "absent when installed from a wheel" behavior this lane's
    design calls for, with no special-casing needed here.
    """
    import macroforecast as _mf

    return Path(_mf.__file__).resolve().parent


def _environment_provenance() -> dict[str, Any]:
    """The git/environment/dependency block, reusing ``output.collect_provenance``.

    Deliberately passes ``cwd=`` pointing at the macroforecast PACKAGE's own
    checkout, not the caller's current working directory (the default
    ``collect_provenance()`` behavior used by the opt-in save path, which is
    left unchanged) -- a referee needs to know which macroforecast commit/build
    produced this report regardless of what directory the analysis script
    itself was run from.
    """
    from macroforecast.output import collect_provenance

    return collect_provenance(cwd=_package_source_root())


def _panel_fingerprint(frame: pd.DataFrame) -> dict[str, Any]:
    """A stable sha256 fingerprint over the panel's index, columns, and values.

    Full content by default (index as int64 ns timestamps, column names in
    order, values as explicit little-endian float64 bytes -- so the digest is
    stable across platforms/byte orders, not just across runs on one machine).
    Above :data:`_FINGERPRINT_FULL_CELL_CAP` cells the digest is computed from
    a deterministic strided subsample instead (same row/col stride every call
    for the same shape), and ``method``/``row_stride``/``col_stride`` record
    this so a referee never mistakes it for a full-content digest.
    """
    n_rows, n_cols = frame.shape
    total_cells = n_rows * n_cols
    row_stride = col_stride = 1
    method = "full_content"
    sampled = frame
    if total_cells > _FINGERPRINT_FULL_CELL_CAP and n_rows > 0 and n_cols > 0:
        reduction = total_cells / _FINGERPRINT_FULL_CELL_CAP
        row_stride = max(1, round(reduction ** 0.5))
        col_stride = max(1, round(reduction / row_stride))
        sampled = frame.iloc[::row_stride, ::col_stride]
        method = "strided_subsample"

    digest = hashlib.sha256()
    try:
        digest.update(np.ascontiguousarray(sampled.index.asi8).tobytes())
    except AttributeError:
        # Non-datetime index (should not happen for a canonical panel, but the
        # fingerprint must never raise): fall back to a stable string form.
        digest.update("\x1f".join(str(v) for v in sampled.index).encode())
    digest.update("\x1f".join(str(c) for c in sampled.columns).encode())
    values = np.ascontiguousarray(sampled.to_numpy(dtype="float64"))
    digest.update(values.astype("<f8", copy=False).tobytes())

    return {
        "algorithm": "sha256",
        "method": method,
        "value": digest.hexdigest(),
        "row_stride": row_stride,
        "col_stride": col_stride,
        "sampled_shape": [int(sampled.shape[0]), int(sampled.shape[1])],
    }


def _data_identity(data: Any) -> dict[str, Any]:
    """Data-identity descriptor: dataset/source_family/vintage (from bundle
    metadata), panel shape/date range/column count, and a content fingerprint.

    Best-effort over any accepted ``data`` shape (``DataBundle``/``DataSpec``/
    ``(panel, metadata)`` tuple/bare ``DataFrame``); a metadata-free or
    malformed input yields ``None`` for the fields that do not resolve rather
    than raising -- provenance collection must never break a run.
    """
    from macroforecast.data import metadata as _bundle_metadata

    try:
        meta = dict(_bundle_metadata(data))
    except Exception:
        meta = {}

    frame = _panel_frame(data)
    n_rows = n_columns = start = end = None
    fingerprint: dict[str, Any] | None = None
    if isinstance(frame, pd.DataFrame):
        n_rows = int(frame.shape[0])
        n_columns = int(frame.shape[1])
        if len(frame.index):
            try:
                start = frame.index[0].isoformat()
                end = frame.index[-1].isoformat()
            except AttributeError:
                start = str(frame.index[0])
                end = str(frame.index[-1])
        try:
            fingerprint = _panel_fingerprint(frame)
        except Exception as exc:  # pragma: no cover - defensive, never break a run
            fingerprint = {
                "algorithm": "sha256",
                "method": "unavailable",
                "error": f"{type(exc).__name__}: {exc}",
            }

    return {
        "dataset": meta.get("dataset"),
        "source_family": meta.get("source_family"),
        "vintage": meta.get("vintage"),
        "frequency": meta.get("frequency"),
        "n_rows": n_rows,
        "n_columns": n_columns,
        "start": start,
        "end": end,
        "fingerprint": fingerprint,
    }


def _spec_echo(spec: PipelineSpec) -> dict[str, Any]:
    """A plain, JSON-able echo of the resolved spec's key choices.

    Not a substitute for the spec itself -- ``preprocessing``/``features``
    objects are not serialized here, only the choices a referee needs to see
    the shape of the design: targets/policies, horizons, window cutoffs,
    arms/models, seed, and worker/cache configuration.
    """
    window = spec.window
    if hasattr(window, "to_dict"):
        try:
            window_echo = window.to_dict()
        except Exception as exc:  # pragma: no cover - defensive
            window_echo = {"repr": str(window), "to_dict_error": f"{type(exc).__name__}: {exc}"}
    else:
        window_echo = {"repr": str(window)}

    def _metric_name(metric: Any) -> str:
        return metric if isinstance(metric, str) else str(getattr(metric, "__name__", metric))

    return {
        "targets": [
            {"name": t.name, "policy": t.policy, "transform": t.transform, "tcode": t.tcode}
            for t in spec.targets
        ],
        "horizons": list(spec.horizons),
        "window": window_echo,
        "arms": [
            {
                "name": a.name,
                "model": _model_default_name(a.model),
                "is_benchmark": bool(a.is_benchmark),
                "nested_in_benchmark": bool(a.nested_in_benchmark),
            }
            for a in spec.arms
        ],
        "benchmark": spec.evaluation.benchmark,
        "evaluation": {
            "metrics": [_metric_name(m) for m in spec.evaluation.metrics],
            "tests": list(spec.evaluation.tests),
            "mcs_alpha": spec.evaluation.mcs_alpha,
            "mcs_method": spec.evaluation.mcs_method,
        },
        "seed": spec.seed,
        "n_jobs": spec.n_jobs,
        "model_threads": spec.model_threads,
        "preprocessing_cache_dir": spec.preprocessing_cache_dir,
        "checkpoint_dir": spec.checkpoint_dir,
    }


def _audit(spec: PipelineSpec) -> tuple[dict, dict]:
    """Collect provenance and a leakage audit for the pipeline execution."""
    import macroforecast as _mf

    provenance: dict = dict(spec.provenance)
    provenance.update({
        "package_version": getattr(_mf, "__version__", "unknown"),
        "seed": spec.seed,
        "targets": [
            {"name": t.name, "policy": t.policy, "transform": t.transform, "tcode": t.tcode}
            for t in spec.targets
        ],
        "horizons": list(spec.horizons),
        "arms": [a.name for a in spec.arms],
        "benchmark": spec.evaluation.benchmark,
        "combinations": [c.name for c in spec.combinations],
    })
    # "basic" (opt-out) reproduces exactly the dict shape above -- the pre-
    # existing behavior. "full" (default) additionally self-certifies the
    # report: WHERE it ran (environment/git), WHAT DATA it ran on (identity +
    # content fingerprint), and WHAT WAS ASKED FOR (a plain echo of the
    # resolved spec's key choices) -- see the module-level comment above.
    if getattr(spec, "provenance_level", "full") == "full":
        provenance["environment"] = _environment_provenance()
        provenance["data"] = _data_identity(spec.data)
        provenance["spec_echo"] = _spec_echo(spec)
    leakage: dict = {}
    # Surface window.validate() warnings (e.g. embargo < horizon-1) when available.
    # The runner injects each horizon into the window's test spec at execution
    # time, so validate the window once PER horizon (not the base test.horizon)
    # to surface the embargo / pseudo-OOS warning that only fires for horizon > 1.
    import dataclasses as _dc

    index = _panel_index(spec.data)
    warnings_seen: list[str] = []
    window_ok = True
    if index is not None and hasattr(spec.window, "validate"):
        for h in spec.horizons:
            try:
                if hasattr(spec.window, "test"):
                    test_spec = _dc.replace(spec.window.test, horizon=int(h))
                    wspec = _dc.replace(spec.window, test=test_spec)
                else:
                    wspec = spec.window
                report = wspec.validate(index)
            except Exception as exc:
                # A validation that crashes must NOT be read as leak-free; record
                # the failure and mark the window not-ok rather than staying silent.
                report = None
                window_ok = False
                msg = f"h={h}: window.validate raised {type(exc).__name__}: {exc}"
                if msg not in warnings_seen:
                    warnings_seen.append(msg)
            if isinstance(report, dict):
                window_ok = window_ok and bool(report.get("ok", True))
                for w in report.get("warnings", []):
                    tagged = w if str(h) in w else f"h={h}: {w}"
                    if tagged not in warnings_seen:
                        warnings_seen.append(tagged)
        leakage["window_warnings"] = warnings_seen
        leakage["window_ok"] = window_ok
    leakage["preprocessing_policies"] = {
        a.name: str(a.preprocessing_policy) for a in spec.arms
    }
    return provenance, leakage


def run_pipeline(spec: PipelineSpec):
    """Execute the full pipeline: run arms, evaluate, and assemble a PipelineReport."""
    from macroforecast.pipeline.evaluate import evaluate
    from macroforecast.pipeline.spec import PipelineReport

    master, failed_cells, empty_cells = _run_cells(spec)
    results = evaluate(master, spec)
    provenance, leakage = _audit(spec)
    # Mirror per-cell failures and zero-row cells into the leakage audit so any
    # consumer that reads only the audit still sees that some arms failed to run or
    # that some (target, horizon) cells silently produced no forecasts.
    leakage = {
        **leakage,
        "failed_cells": list(failed_cells),
        "empty_cells": list(empty_cells),
    }
    return PipelineReport(
        forecasts=results["forecasts"],
        accuracy=results["accuracy"],
        significance=results["significance"],
        mcs=results["mcs"],
        provenance=provenance,
        leakage_audit=leakage,
        interpretation=None,
        model_store=spec.model_store,
        spec=spec,
        failed_cells=tuple(failed_cells),
        empty_cells=tuple(empty_cells),
        density=results["density"],
        calibration=results["calibration"],
    )
