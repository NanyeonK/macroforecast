"""A2 second half — one resolver for one forecast task.

Before this, "which task is this cell" was resolved twice. `pipeline/run.py` retargeted
the arm's `FeatureSpec` and **raised** on failure; `pipeline/result_store.py` retargeted
it again and **swallowed** the failure, digesting the un-retargeted spec. Execution and
cache identity therefore disagreed about what a failed retarget means.

In practice execution raised first, so the bad digest was never used. "In practice the
wrong one never runs" is not a contract, and these tests make it one.
"""
from __future__ import annotations

import dataclasses as dc

import pytest

from macroforecast.forecasting.task import (
    FeatureRetargetError,
    effective_target,
    resolve_forecast_task,
    retarget_features,
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
