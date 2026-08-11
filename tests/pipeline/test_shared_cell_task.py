"""One resolved task per cell, shared by every consumer that needs to know the cell.

Execution, the result-store digest, the checkpoint cell manifest and the provenance
echo all have to answer "which forecast is this". They used to answer it separately,
and two of the answers disagreed about what an unretargetable feature spec means (the
execution path raised; the identity path swallowed it and digested the un-retargeted
spec). The pipeline now resolves once per cell and hands the same
`ResolvedForecastTask` objects to all of them.

The invariant that makes the change safe is that the digest must NOT MOVE: an existing
result store stays reusable. So most of these tests are equality against the digest the
spec-only spelling produces.
"""
from __future__ import annotations

import dataclasses as dc
import json
import pickle

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.forecasting.task import FeatureRetargetError
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline
from macroforecast.pipeline.result_store import result_cell_identity
from macroforecast.pipeline.run import _cell_tasks, _data_identity

N = 72
TEST_START = 54


@pytest.fixture(scope="module")
def panel():
    idx = pd.date_range("1998-01-31", periods=N, freq="ME", name="date")
    rng = np.random.default_rng(23)
    frame = pd.DataFrame({f"x{i}": rng.normal(size=N) for i in range(3)}, index=idx)
    frame["y"] = 0.4 * frame["x0"] + rng.normal(size=N) * 0.3
    frame["z"] = -0.5 * frame["x1"] + rng.normal(size=N) * 0.3
    bundle = mf.data.custom_dataset(frame, transform_codes={c: 1 for c in frame.columns})
    return idx, bundle


def _features(target="SOMETHING_ELSE"):
    """Deliberately pointed at another series, so the retarget is never a no-op."""
    return mf.feature_engineering.feature_spec(
        target=target, predictors=["x0", "x1"], lags=0, target_lags=2
    )


class _UncomparableModel:
    """A model object whose equality answer is unusable, like an array-valued `__eq__`."""

    def __eq__(self, other):
        raise TypeError("this object refuses to be compared")

    __hash__ = object.__hash__


def _spec(idx, bundle, *, horizons=(1, 2), arms=None, targets=("y", "z"), **kw):
    arms = arms or [Arm("OLS", model="ols", features=_features(), is_benchmark=True)]
    return pipeline_spec(
        data=bundle,
        targets=[TargetSpec(name, transform="level") for name in targets],
        horizons=list(horizons),
        window=mf.window.from_cutoffs(
            test_start=idx[TEST_START],
            horizon=max(horizons),
            embargo=0,
            val_method="expanding",
            val_min_train_size=20,
        ),
        arms=arms,
        evaluation=EvalSpec(benchmark=arms[0].name, metrics=("rmse",), tests=()),
        save_models=False,
        n_jobs=1,
        seed=5,
        **kw,
    )


# --------------------------------------------------------------------------- #
# The digest does not move
# --------------------------------------------------------------------------- #

def test_passing_the_task_produces_the_identical_digest_and_echo(panel):
    """Sharing the resolution must not invalidate a single cached cell.

    If the task-driven digest differed from the spec-only digest, every existing result
    store would silently miss on its next run and recompute the whole study.
    """
    idx, bundle = panel
    spec = _spec(idx, bundle)
    identity_data = _data_identity(spec.data)
    for target in spec.targets:
        tasks = _cell_tasks(spec, spec.arms[0], target, (1, 2))
        for task in tasks:
            with_task = result_cell_identity(
                spec, spec.arms[0], task.target, horizon=task.horizon,
                data_identity=identity_data, task=task,
            )
            without_task = result_cell_identity(
                spec, spec.arms[0], task.target, horizon=task.horizon,
                data_identity=identity_data,
            )
            assert with_task.digest == without_task.digest, (
                f"{target.name}/h{task.horizon}: the digest moved when the resolved "
                "task was shared instead of re-derived"
            )
            assert with_task.cell_echo == without_task.cell_echo
            assert with_task.digest is not None


def test_the_echo_describes_the_retargeted_features_not_the_arms_own_spec(panel):
    """The arm's spec names another series; the digest must describe the cell's."""
    idx, bundle = panel
    spec = _spec(idx, bundle)
    task = _cell_tasks(spec, spec.arms[0], spec.targets[0], (1,))[0]
    identity = result_cell_identity(
        spec, spec.arms[0], task.target, horizon=1,
        data_identity=_data_identity(spec.data), task=task,
    )
    assert identity.cell_echo is not None
    assert identity.cell_echo["arm"]["features"]["spec"]["target"] == "y"
    assert spec.arms[0].features.target == "SOMETHING_ELSE", "the spec itself is untouched"


# --------------------------------------------------------------------------- #
# One resolution per cell, counted over a production run
# --------------------------------------------------------------------------- #

def _count_retargets(monkeypatch):
    """Record every `retarget_features` call made anywhere, in call order.

    Patching the definition module catches BOTH callers: `resolve_forecast_tasks`
    reaches it as a module global, and `result_cell_identity`'s spec-only fallback
    imports it from the module at call time.
    """
    import macroforecast.forecasting.task as task_mod

    calls: list[tuple[str, str]] = []
    real = task_mod.retarget_features

    def counting(features, target_name, *, arm_name=""):
        calls.append((arm_name, target_name))
        return real(features, target_name, arm_name=arm_name)

    monkeypatch.setattr(task_mod, "retarget_features", counting)
    return calls


def test_store_preflight_and_execution_share_one_retarget_per_cell(panel, tmp_path, monkeypatch):
    """The regression this refactor closes, stated as a count with the store ENABLED.

    Store identity is computed for every cell before any cell runs, and execution then
    runs the cells that missed. Both need the cell's resolved features. When each
    resolved for itself, every successful cell retargeted TWICE and the two consumers
    held two different feature objects -- the digest describing one and the run using
    the other, agreeing only because the same deterministic function produced both.
    That is the coincidence the shared task removes, so the count is the contract.

    The count is per CELL, not per horizon and not per consumer: two targets, one arm,
    one horizon each is two cells and therefore two retargets.
    """
    idx, bundle = panel
    calls = _count_retargets(monkeypatch)
    spec = _spec(idx, bundle, horizons=(1,), result_store=str(tmp_path / "store"))

    report = run_pipeline(spec)

    assert list(report.failed_cells) == [], f"the run failed: {list(report.failed_cells)}"
    assert report.provenance["result_store"]["n_computed"] == 2
    assert sorted(calls) == [("OLS", "y"), ("OLS", "z")], (
        f"expected one retarget per successful cell, got {len(calls)}: {calls}. "
        "A count of two per cell means the store preflight and execution each resolved "
        "the cell, so they no longer share one answer."
    )

    calls.clear()
    reused = run_pipeline(spec)
    assert reused.provenance["result_store"]["n_reused"] == 2
    assert sorted(calls) == [("OLS", "y"), ("OLS", "z")], (
        f"a fully reused run must still resolve each cell once, got {calls}"
    )


@pytest.mark.parametrize("with_store", [False, True], ids=["grouped", "split-by-store"])
def test_the_count_is_per_cell_under_both_horizon_groupings(
    panel, tmp_path, monkeypatch, with_store
):
    """One retarget per CELL, whichever way the horizons are grouped.

    Enabling the result store is also a grouping decision: `_enumerate_cells` splits one
    horizon per cell so the persisted unit is exactly one (target, horizon, arm), while
    the storeless serial path groups all three horizons into one cell. The contract is
    the same either way and the arithmetic differs, which is precisely why it is worth
    counting both -- three cells resolve three times, one grouped cell resolves once,
    and neither resolves twice per cell.
    """
    idx, bundle = panel
    calls = _count_retargets(monkeypatch)
    store = {"result_store": str(tmp_path / "store")} if with_store else {}
    spec = _spec(idx, bundle, horizons=(1, 2, 3), targets=("y",), **store)

    report = run_pipeline(spec)

    assert list(report.failed_cells) == [], f"the run failed: {list(report.failed_cells)}"
    assert sorted(report.forecasts["horizon"].unique().tolist()) == [1, 2, 3], (
        "resolving once must not cost a horizon"
    )
    expected = 3 if with_store else 1
    assert calls == [("OLS", "y")] * expected, (
        f"expected {expected} retarget(s) for this grouping, got {len(calls)}: {calls}"
    )


def test_the_store_preflight_and_execution_receive_the_same_task_objects(
    panel, tmp_path, monkeypatch
):
    """Not merely equal answers -- the SAME tuple, so they cannot drift by construction.

    Equal-but-separate objects would still pass the count above if a future change
    resolved once and copied; identity is what makes "the digest describes the forecast
    that ran" structural rather than a property of the resolver being deterministic.
    """
    from macroforecast.pipeline import run as run_mod

    preflight: dict = {}
    executed: dict = {}
    real_load = run_mod._result_store_load
    real_execute = run_mod._execute_cell

    def recording_load(spec, cell, store, data_identity, metadata, tasks=None):
        preflight[cell] = tasks
        return real_load(spec, cell, store, data_identity, metadata, tasks)

    def recording_execute(spec, cell, *, preprocessing_cache=None, tasks=None):
        executed[cell] = tasks
        return real_execute(spec, cell, preprocessing_cache=preprocessing_cache, tasks=tasks)

    monkeypatch.setattr(run_mod, "_result_store_load", recording_load)
    monkeypatch.setattr(run_mod, "_execute_cell", recording_execute)

    idx, bundle = panel
    spec = _spec(idx, bundle, horizons=(1,), result_store=str(tmp_path / "store"))
    report = run_pipeline(spec)

    assert list(report.failed_cells) == []
    assert executed and set(executed) == set(preflight)
    for cell, tasks in executed.items():
        assert tasks is not None, "execution was handed no pre-resolved task"
        assert preflight[cell] is tasks, (
            "the store preflight and execution hold different task tuples for one cell"
        )
        assert tasks[0].features is preflight[cell][0].features


# --------------------------------------------------------------------------- #
# The parallel payload carries the resolution across the process boundary
# --------------------------------------------------------------------------- #

def test_the_parallel_worker_consumes_the_payload_tasks_instead_of_resolving(
    panel, monkeypatch
):
    """The parallel half of the same contract, without spawning a pool.

    Calling the module-level worker directly is what lets the retarget count be
    observed at all: a real subprocess would do its resolving in another interpreter
    where the counter does not exist, so a worker that re-resolved would look identical
    to one that did not.
    """
    from macroforecast.pipeline import run as run_mod

    idx, bundle = panel
    spec = _spec(idx, bundle, horizons=(1,), targets=("y",))
    cell = run_mod._enumerate_cells(spec)[0]
    tasks = _cell_tasks(spec, spec.arms[cell.arm_idx], spec.targets[cell.target_idx], cell.horizons)
    payload = (dc.replace(spec, data=None), cell, "token-for-this-test", tasks)

    # The payload crosses a process boundary, so everything newly added to it must
    # pickle -- and the spec and its tasks must survive as ONE pickle, keeping the
    # sharing that makes the arm's model and the task's model the same object.
    restored_spec, _, _, restored_tasks = pickle.loads(pickle.dumps(payload))
    assert restored_tasks[0].model is restored_spec.arms[0].model

    monkeypatch.setitem(run_mod._WORKER_DATA_BY_TOKEN, "token-for-this-test", spec.data)
    calls = _count_retargets(monkeypatch)
    returned_cell, frame, error = run_mod._parallel_cell_worker(payload)

    assert error is None, f"the worker failed: {error}"
    assert returned_cell == cell
    assert frame is not None and not frame.empty
    assert calls == [], (
        f"the worker re-resolved a cell the parent had already resolved: {calls}"
    )


def test_a_worker_without_tasks_still_fails_only_its_own_cell(panel):
    """`None` tasks (an unresolvable cell) must stay a returned error, not an escape."""
    from macroforecast.pipeline import run as run_mod

    idx, bundle = panel
    broken = Arm("BAD", model="ols", features=_Unretargetable(), is_benchmark=True)
    spec = _spec(idx, bundle, horizons=(1,), targets=("y",), arms=[broken])
    cell = run_mod._enumerate_cells(spec)[0]
    # None is what the parent puts in the payload for a cell it could not resolve.
    payload = (dc.replace(spec, data=None), cell, "token-for-broken-cell", None)

    run_mod._WORKER_DATA_BY_TOKEN["token-for-broken-cell"] = spec.data
    try:
        returned_cell, frame, error = run_mod._parallel_cell_worker(payload)
    finally:
        run_mod._WORKER_DATA_BY_TOKEN.pop("token-for-broken-cell", None)

    assert returned_cell == cell and frame is None
    assert "could not re-target feature spec" in str(error)


# --------------------------------------------------------------------------- #
# A task for a different cell is a caller bug, not an uncacheable cell
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "wrong, named",
    [
        (dict(horizon=9), "horizon"),
        (dict(arm_name="OTHER_ARM"), "arm"),
        # The model is digested, so a task naming another one would mint this arm's
        # digest for a different arm's fit -- the divergence the runner also refuses.
        (dict(model="ridge"), "model"),
    ],
)
def test_a_task_for_another_cell_is_refused(panel, wrong, named):
    idx, bundle = panel
    spec = _spec(idx, bundle)
    task = _cell_tasks(spec, spec.arms[0], spec.targets[0], (1,))[0]
    with pytest.raises(ValueError) as exc:
        result_cell_identity(
            spec, spec.arms[0], task.target, horizon=1,
            data_identity=_data_identity(spec.data), task=dc.replace(task, **wrong),
        )
    assert named in str(exc.value)
    assert "different cell" in str(exc.value)


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(None, id="no-model"),
        pytest.param(_UncomparableModel(), id="exotic-eq"),
    ],
)
def test_a_task_the_model_check_cannot_judge_is_still_digested(panel, model):
    """The model guard is one-sided, so it cannot break a caller it was not aimed at.

    A task carrying no model states nothing to contradict (that is how the runner reads
    it too), and a model whose ``__eq__`` raises or answers with something that is not a
    boolean is no evidence of a different arm. Either would be a poor reason to refuse a
    digest for a cell that is otherwise perfectly identified.
    """
    idx, bundle = panel
    spec = _spec(idx, bundle)
    task = _cell_tasks(spec, spec.arms[0], spec.targets[0], (1,))[0]
    identity = result_cell_identity(
        spec, spec.arms[0], task.target, horizon=1,
        data_identity=_data_identity(spec.data), task=dc.replace(task, model=model),
    )
    assert identity.digest is not None


def test_a_task_for_another_target_is_refused(panel):
    """The digest would then be filed under one target and hold another's forecast."""
    idx, bundle = panel
    spec = _spec(idx, bundle)
    y_task = _cell_tasks(spec, spec.arms[0], spec.targets[0], (1,))[0]
    z_target = spec.targets[1]
    with pytest.raises(ValueError, match="different cell"):
        result_cell_identity(
            spec, spec.arms[0], z_target, horizon=1,
            data_identity=_data_identity(spec.data), task=y_task,
        )


# --------------------------------------------------------------------------- #
# Checkpoint identity comes from the same tasks
# --------------------------------------------------------------------------- #

def test_every_horizon_gets_its_own_checkpoint_manifest_from_the_shared_task(panel, tmp_path):
    """Per-horizon checkpoint paths, per-horizon digests, one resolution behind them.

    The echo is compared field by field against the shared task rather than against a
    digest recomputed here: ``effective_selection_seed`` reads the package config, which
    ``run_pipeline`` sets from ``spec.seed`` for the duration of the run, so a digest
    computed outside a run legitimately differs. The digests' reproducibility is pinned
    by the second run below instead.
    """
    idx, bundle = panel
    ckpt = tmp_path / "ckpt"
    spec = _spec(idx, bundle, horizons=(1, 2, 3), checkpoint_dir=str(ckpt))
    report = run_pipeline(spec)
    assert list(report.failed_cells) == []

    digests = {}
    for target in spec.targets:
        for task in _cell_tasks(spec, spec.arms[0], target, (1, 2, 3)):
            path = ckpt / f"{target.name}__OLS" / f"h{task.horizon}" / "cell_manifest.json"
            assert path.exists(), f"missing manifest for {target.name}/h{task.horizon}"
            manifest = json.loads(path.read_text())
            echo = manifest["cell_echo"]
            assert manifest["horizon"] == task.horizon
            assert manifest["target"] == target.name
            assert manifest["effective_target"] == task.target_name
            assert manifest["undigestible_reason"] is None
            assert echo["horizon"] == task.horizon
            assert echo["target"]["name"] == task.target_name
            assert echo["target"]["forecast_policy"] == task.forecast_policy
            assert echo["target"]["transform"] == task.target_transform
            # The retargeted features -- the half that used to be derived separately
            # here and in the execution path.
            assert echo["arm"]["features"]["spec"]["target"] == task.features.target
            digests[(target.name, task.horizon)] = manifest["digest"]

    assert len(set(digests.values())) == len(digests), (
        f"two cells share a digest, so one would serve the other's forecast: {digests}"
    )

    rerun_ckpt = tmp_path / "ckpt_again"
    run_pipeline(_spec(idx, bundle, horizons=(1, 2, 3), checkpoint_dir=str(rerun_ckpt)))
    for (target_name, horizon), digest in digests.items():
        again = json.loads(
            (rerun_ckpt / f"{target_name}__OLS" / f"h{horizon}" / "cell_manifest.json")
            .read_text()
        )
        assert again["digest"] == digest, (
            f"{target_name}/h{horizon}: the checkpoint digest is not reproducible"
        )


def test_a_policy_override_reaches_both_the_manifest_and_the_forecast(panel, tmp_path):
    """`effective_target` differs from the spec target only in policy.

    The checkpoint DIRECTORY is named for the spec target (so a rerouted cell keeps its
    path) while the digest and the forecast rows must both carry the overridden policy.
    """
    idx, bundle = panel
    ckpt = tmp_path / "ckpt_override"
    arms = [
        Arm("OLS", model="ols", features=_features(), is_benchmark=True),
        Arm("NAIVE", model="naive", features=None),
    ]
    spec = _spec(
        idx, bundle, horizons=(1, 2), arms=arms, checkpoint_dir=str(ckpt),
        on_unsupported_direct="reroute",
    )
    assert spec.policy_overrides, "the fixture must actually produce an override"
    report = run_pipeline(spec)
    assert list(report.failed_cells) == []

    naive_rows = report.forecasts[report.forecasts["arm"] == "NAIVE"]
    assert set(naive_rows["forecast_policy"].unique()) == {"recursive"}

    manifest = json.loads(
        (ckpt / "y__NAIVE" / "h2" / "cell_manifest.json").read_text()
    )
    assert manifest["target"] == "y", "the path target names the directory"
    assert manifest["cell_echo"]["target"]["forecast_policy"] == "recursive", (
        "the digest must describe the policy the cell actually ran under"
    )


# --------------------------------------------------------------------------- #
# An unresolvable cell stays a per-cell failure
# --------------------------------------------------------------------------- #

@dc.dataclass(frozen=True)
class _Unretargetable:
    """A feature spec `dataclasses.replace` cannot re-point."""

    target: str = "OTHER"
    targets: tuple = ()

    def __post_init__(self):
        if self.target != "OTHER":
            raise TypeError("cannot be retargeted")


def test_an_unresolvable_cell_has_no_tasks(panel):
    idx, bundle = panel
    spec = _spec(idx, bundle)
    broken = dc.replace(spec.arms[0], features=_Unretargetable())
    with pytest.raises(FeatureRetargetError, match="OLS"):
        _cell_tasks(dc.replace(spec, arms=(broken,)), broken, spec.targets[0], (1, 2))


def test_an_unresolvable_cell_is_undigestible_and_fails_only_its_own_cell(panel, tmp_path):
    """Resolving up front must not turn a per-cell failure into an aborted run.

    Store identity is computed for EVERY cell before any cell executes, outside the
    per-cell error handling. If the resolution raised there, one bad arm would take the
    whole study down instead of appearing in `failed_cells` beside the arms that ran.
    """
    idx, bundle = panel
    good = Arm("GOOD", model="ols", features=_features(), is_benchmark=True)
    bad = Arm("BAD", model="ols", features=_Unretargetable())
    spec = _spec(
        idx, bundle, horizons=(1,), arms=[good, bad],
        result_store=str(tmp_path / "store"),
    )
    with pytest.warns(RuntimeWarning, match="result_store cannot digest cell"):
        report = run_pipeline(spec)

    failed = {(row["arm"], row["target"]) for row in report.failed_cells}
    assert failed == {("BAD", "y"), ("BAD", "z")}, f"unexpected failures: {failed}"
    assert all(
        "could not re-target feature spec" in row["error"] for row in report.failed_cells
    )
    assert set(report.forecasts["arm"].unique()) == {"GOOD"}, (
        "the arms that could be resolved must still have run"
    )
    store_metadata = report.provenance["result_store"]
    assert store_metadata["n_undigestible"] == 2
    assert store_metadata["n_computed"] == 2


# --------------------------------------------------------------------------- #
# End to end: the shared resolution keeps the store reusable
# --------------------------------------------------------------------------- #

def test_a_retargeted_arm_round_trips_through_the_result_store(panel, tmp_path):
    idx, bundle = panel
    spec = _spec(idx, bundle, horizons=(1, 2), result_store=str(tmp_path / "store"))
    first = run_pipeline(spec)
    second = run_pipeline(spec)

    assert first.provenance["result_store"]["n_computed"] == 4
    assert first.provenance["result_store"]["n_reused"] == 0
    assert second.provenance["result_store"]["n_reused"] == 4
    assert second.provenance["result_store"]["n_computed"] == 0

    sort_by = ["target", "horizon", "arm", "date"]
    pd.testing.assert_frame_equal(
        first.forecasts.sort_values(sort_by).reset_index(drop=True),
        second.forecasts.sort_values(sort_by).reset_index(drop=True),
    )
