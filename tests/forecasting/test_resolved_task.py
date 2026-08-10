"""A2 second half — one resolver for one forecast task.

Before this, "which task is this cell" was resolved twice. `pipeline/run.py` retargeted
the arm's `FeatureSpec` and **raised** on failure; `pipeline/result_store.py` retargeted
it again and **swallowed** the failure, digesting the un-retargeted spec. Execution and
cache identity therefore disagreed about what a failed retarget means.

In practice execution raised first, so the bad digest was never used. "In practice the
wrong one never runs" is not a contract, and these tests make it one.

The later sections cover the horizon-GROUP resolver, which is what makes "resolve once"
structural rather than a convention: a cell answers "which series, under which policy,
from which features" once and then stamps every horizon it was asked for, and a sequence
of tasks that disagrees about any of that is refused rather than executed under the first
task's design.
"""
from __future__ import annotations

import dataclasses as dc

import pytest

from macroforecast.forecasting.task import (
    FeatureRetargetError,
    ResolvedForecastTask,
    effective_target,
    resolve_forecast_task,
    resolve_forecast_tasks,
    retarget_features,
    validate_task_sequence,
)


@dc.dataclass(frozen=True)
class _Target:
    name: str
    policy: str = "direct"
    transform: str = "level"


@dc.dataclass(frozen=True)
class _Features:
    target: str
    targets: tuple[str, ...] = ()


@dc.dataclass(frozen=True)
class _Arm:
    name: str
    features: object = None
    model: str = "ols"


@dc.dataclass(frozen=True)
class _Spec:
    policy_overrides: dict = dc.field(default_factory=dict)


def test_the_resolver_needs_no_pipeline_types():
    """A forecast task is a statement about one forecast.

    The fixtures above are plain local dataclasses, not `pipeline.spec` objects. That
    they work is the property: `forecasting` sits below `pipeline`, and the
    architecture guard rejected an earlier draft of `task.py` that imported pipeline
    types even under TYPE_CHECKING. Structural parameters are what let a direct
    `forecasting.run()` reach the same resolver the pipeline uses.
    """
    task = resolve_forecast_task(
        _Spec(), _Arm("A", features=_Features(target="y")), _Target("y"), horizon=3
    )
    assert task.target_name == "y"
    assert task.horizon == 3
    assert task.forecast_policy == "direct"
    assert task.target_transform == "level"


def test_a_policy_override_reaches_the_task():
    spec = _Spec(policy_overrides={("A", "y"): "recursive"})
    task = resolve_forecast_task(spec, _Arm("A"), _Target("y", policy="direct"), horizon=1)
    assert task.forecast_policy == "recursive"


def test_an_override_for_another_arm_does_not():
    spec = _Spec(policy_overrides={("B", "y"): "recursive"})
    task = resolve_forecast_task(spec, _Arm("A"), _Target("y", policy="direct"), horizon=1)
    assert task.forecast_policy == "direct"


def test_the_target_object_is_not_mutated_when_no_override_applies():
    """No override means the same object, not a copy: identity is cheap to check."""
    target = _Target("y", policy="direct")
    assert effective_target(_Spec(), _Arm("A"), target) is target


def test_features_are_retargeted_to_the_cell_target():
    """A multi-target pipeline runs each arm for every target."""
    out = retarget_features(_Features(target="OTHER"), "y", arm_name="A")
    assert out.target == "y"


def test_a_multi_target_feature_spec_is_collapsed():
    out = retarget_features(_Features(target="y", targets=("y", "z")), "y", arm_name="A")
    assert out.target == "y"
    assert out.targets == ()


def test_features_already_on_target_are_returned_unchanged():
    features = _Features(target="y")
    assert retarget_features(features, "y") is features


def test_a_failed_retarget_raises_rather_than_returning_the_wrong_spec():
    """The divergence this step removes.

    `run.py` raised here and `result_store.py` returned the ORIGINAL spec. A feature
    spec still pointing at another target would forecast the wrong series, and a digest
    computed from it would describe a task that was never run. One behaviour now, and
    it is the loud one.
    """

    class _Unreplaceable:
        target = "OTHER"
        targets = ()

    with pytest.raises(FeatureRetargetError) as exc:
        retarget_features(_Unreplaceable(), "y", arm_name="ARM")
    assert "ARM" in str(exc.value), "the error must name the arm a user has to fix"
    assert "y" in str(exc.value), "and the target it could not reach"


def test_none_features_stay_none():
    assert retarget_features(None, "y") is None
    assert resolve_forecast_task(_Spec(), _Arm("A"), _Target("y"), horizon=1).features is None


# --------------------------------------------------------------------------- #
# A horizon GROUP resolves the design once and stamps every horizon
# --------------------------------------------------------------------------- #

def test_a_horizon_group_resolves_the_design_once_and_keeps_every_horizon():
    """The property that makes "resolve once" structural rather than a convention.

    The policy override and the feature retarget answer "which series, under which
    policy, from which features" -- none of which depends on the horizon. A cell that
    forecasts three horizons must therefore resolve once and stamp three horizons, and
    it must not lose one along the way.
    """
    spec = _Spec(policy_overrides={("A", "y"): "recursive"})
    arm = _Arm("A", features=_Features(target="OTHER"))
    tasks = resolve_forecast_tasks(spec, arm, _Target("y", policy="direct"), (1, 3, 6))

    assert [task.horizon for task in tasks] == [1, 3, 6], "every requested horizon survives"
    # One resolution: the horizon-independent parts are the SAME OBJECTS, not merely
    # equal copies. Sharing the object is what a per-horizon re-resolution cannot fake.
    assert len({id(task.features) for task in tasks}) == 1
    assert len({id(task.target) for task in tasks}) == 1
    assert all(task.features.target == "y" for task in tasks)
    assert all(task.forecast_policy == "recursive" for task in tasks)


def test_the_single_horizon_resolver_is_the_group_resolver():
    """Two spellings, one code path, so the two cannot drift."""
    spec = _Spec(policy_overrides={("A", "y"): "recursive"})
    arm = _Arm("A", features=_Features(target="OTHER"))
    single = resolve_forecast_task(spec, arm, _Target("y"), horizon=4)
    (grouped,) = resolve_forecast_tasks(spec, arm, _Target("y"), (4,))
    assert single == grouped


def test_a_cell_with_no_horizons_is_refused():
    with pytest.raises(ValueError, match="at least one horizon"):
        resolve_forecast_tasks(_Spec(), _Arm("A"), _Target("y"), ())


def test_a_failed_retarget_produces_no_tasks_at_all():
    """Not "some horizons resolved and some not" -- an unresolvable cell has no tasks."""

    class _Unreplaceable:
        target = "OTHER"
        targets = ()

    with pytest.raises(FeatureRetargetError):
        resolve_forecast_tasks(_Spec(), _Arm("A", features=_Unreplaceable()), _Target("y"), (1, 2))


# --------------------------------------------------------------------------- #
# A task sequence is ONE cell
# --------------------------------------------------------------------------- #

def _task(**kw):
    base = dict(
        target=_Target("y"),
        horizon=1,
        features=_Features(target="y"),
        arm_name="A",
        model="ols",
    )
    base.update(kw)
    return ResolvedForecastTask(**base)


def test_a_valid_sequence_reports_its_shared_design_and_every_horizon():
    tasks = resolve_forecast_tasks(_Spec(), _Arm("A"), _Target("y"), (2, 5))
    shared, horizons = validate_task_sequence(tasks)
    assert horizons == (2, 5), "no horizon may be dropped or collapsed"
    assert shared is tasks[0]


def test_the_sequence_order_is_preserved_rather_than_sorted():
    """Uniqueness and ordering are the runner's existing horizon validation's job.

    Sorting here would mean the task path and the loose ``horizons=`` path refuse
    different inputs, which is exactly the kind of quiet divergence being removed.
    """
    _, horizons = validate_task_sequence([_task(horizon=6), _task(horizon=2)])
    assert horizons == (6, 2)


@pytest.mark.parametrize(
    "field, other",
    [
        ("target", _Target("z")),
        ("features", _Features(target="z")),
        ("arm_name", "B"),
        ("model", "ridge"),
    ],
)
def test_a_mixed_sequence_is_refused(field, other):
    """A sequence's consumer applies ONE design to every horizon.

    A mixed sequence would therefore forecast the first task's design and label the
    rows with the others', which is a wrong number rather than an error.
    """
    with pytest.raises(ValueError) as exc:
        validate_task_sequence([_task(horizon=1), _task(horizon=2, **{field: other})])
    assert "ONE cell" in str(exc.value)


def test_a_sequence_mixing_forecast_policy_is_refused():
    with pytest.raises(ValueError, match="forecast_policy"):
        validate_task_sequence(
            [_task(horizon=1), _task(horizon=2, target=_Target("y", policy="recursive"))]
        )


def test_a_sequence_mixing_target_transform_is_refused():
    with pytest.raises(ValueError, match="target_transform"):
        validate_task_sequence(
            [_task(horizon=1), _task(horizon=2, target=_Target("y", transform="growth"))]
        )


def test_an_empty_sequence_is_refused():
    with pytest.raises(ValueError, match="at least one task"):
        validate_task_sequence([])


def test_equivalent_targets_that_are_not_identical_are_accepted():
    """The refusal is about the FORECAST, not about object identity.

    Two structurally equivalent targets describe the same forecast; refusing them
    would make the guard reject correct callers for a reason unrelated to forecasting.
    """
    shared, horizons = validate_task_sequence(
        [_task(horizon=1, target=_Target("y")), _task(horizon=2, target=_Target("y"))]
    )
    assert shared.target_name == "y"
    assert horizons == (1, 2)
