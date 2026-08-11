"""`forecasting.run(task=...)` / `run(tasks=...)`: the pre-resolved runner entry.

The point of the task argument is that ONE resolution drives execution, cache identity,
checkpoint identity and provenance. That only holds if a task-driven call is
indistinguishable from the equivalent loose-keyword call -- otherwise unifying the
resolvers would have quietly changed the numbers -- and if supplying both a task and a
disagreeing keyword is refused rather than reconciled behind the caller's back.

The fixtures build tasks from plain local dataclasses, not `pipeline.spec` objects:
`forecasting` sits below `pipeline`, so a direct `forecasting.run()` must be able to
reach the same resolver the pipeline uses without importing anything from above it.
"""
from __future__ import annotations

import dataclasses as dc

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.forecasting.task import resolve_forecast_tasks

N = 84
TEST_START = 64


@dc.dataclass(frozen=True)
class _Target:
    name: str
    policy: str = "direct"
    transform: str = "level"


@dc.dataclass(frozen=True)
class _Arm:
    name: str
    features: object = None
    model: object = "ols"


@dc.dataclass(frozen=True)
class _Spec:
    policy_overrides: dict = dc.field(default_factory=dict)


@pytest.fixture(scope="module")
def panel():
    idx = pd.date_range("1995-01-31", periods=N, freq="ME", name="date")
    rng = np.random.default_rng(19)
    frame = pd.DataFrame({f"x{i}": rng.normal(size=N) for i in range(3)}, index=idx)
    frame["y"] = 0.6 * frame["x0"] - 0.2 * frame["x1"] + rng.normal(size=N) * 0.25
    bundle = mf.data.custom_dataset(frame, transform_codes={c: 1 for c in frame.columns})
    return idx, bundle


def _features(target="y"):
    return mf.feature_engineering.feature_spec(
        target=target, predictors=["x0", "x1", "x2"], lags=0, target_lags=2
    )


def _window(idx, horizon):
    return mf.window.from_cutoffs(
        test_start=idx[TEST_START],
        horizon=horizon,
        embargo=0,
        val_method="expanding",
        val_min_train_size=20,
    )


def _tasks(horizons, *, feature_target="OTHER", policy=None, features=None):
    """Tasks for one cell, resolved exactly as the pipeline resolves them."""
    overrides = {("A", "y"): policy} if policy is not None else {}
    return resolve_forecast_tasks(
        _Spec(policy_overrides=overrides),
        _Arm(
            "A",
            features=_features(feature_target) if features is None else features,
            model="ols",
        ),
        _Target("y", transform="level"),
        horizons,
    )


def _loose(bundle, idx, *, horizons, **kw):
    return mf.forecasting.run(
        bundle,
        "ols",
        window=_window(idx, max(horizons)),
        features=_features("y"),
        target="y",
        horizons=list(horizons),
        target_transform="level",
        save_models=False,
        **kw,
    )


def _assert_same_forecasts(left, right, what):
    a = left.forecasts.sort_values(["horizon", "date"]).reset_index(drop=True)
    b = right.forecasts.sort_values(["horizon", "date"]).reset_index(drop=True)
    assert list(a.columns) == list(b.columns), what
    pd.testing.assert_frame_equal(a, b, check_exact=True, obj=what)


# --------------------------------------------------------------------------- #
# Equivalence: a task-driven call IS the loose-keyword call
# --------------------------------------------------------------------------- #

def test_a_single_horizon_task_matches_the_loose_keyword_call(panel):
    idx, bundle = panel
    (task,) = _tasks((2,))
    from_task = mf.forecasting.run(
        bundle, "ols", window=_window(idx, 2), save_models=False, task=task
    )
    _assert_same_forecasts(
        from_task, _loose(bundle, idx, horizons=(2,)), "task= vs loose keywords"
    )


def test_a_horizon_group_task_sequence_matches_the_loose_keyword_call(panel):
    """Serial grouped multi-horizon execution: one call, every horizon, same numbers."""
    idx, bundle = panel
    tasks = _tasks((1, 2, 3))
    from_tasks = mf.forecasting.run(
        bundle, "ols", window=_window(idx, 3), save_models=False, tasks=tasks
    )
    loose = _loose(bundle, idx, horizons=(1, 2, 3))
    _assert_same_forecasts(from_tasks, loose, "tasks= vs loose horizons=")
    assert from_tasks.metadata["run"]["horizons"] == [1, 2, 3]
    assert from_tasks.metadata["run"]["multi_horizon"] is True
    assert sorted(from_tasks.metadata["per_horizon"]) == ["1", "2", "3"], (
        "every horizon must keep its own per-horizon metadata block"
    )


def test_the_task_carries_the_retargeted_features_into_the_run(panel):
    """The arm's spec targeted another series; the task is what re-points it.

    If the runner ignored the task's features and fell back to the arm's own spec, it
    would forecast the wrong series -- silently, since the row labels come from the
    resolved target either way.
    """
    idx, bundle = panel
    (task,) = _tasks((1,), feature_target="SOMETHING_ELSE")
    assert task.features.target == "y"
    result = mf.forecasting.run(
        bundle, "ols", window=_window(idx, 1), save_models=False, task=task
    )
    assert set(result.forecasts["target"].unique()) == {"y"}
    _assert_same_forecasts(result, _loose(bundle, idx, horizons=(1,)), "retargeted features")


def test_a_policy_override_on_the_task_reaches_the_run(panel):
    """The override is a property of the resolved task, not of a keyword at the call
    site: nothing in this call says "recursive" except the task itself."""
    idx, bundle = panel
    autoregressive = mf.feature_engineering.feature_spec(
        target="OTHER", predictors=[], lags=0, target_lags=(0, 1, 2)
    )
    (task,) = _tasks((2,), policy="recursive", features=autoregressive)
    assert task.forecast_policy == "recursive"
    result = mf.forecasting.run(
        bundle, "ols", window=_window(idx, 2), save_models=False, task=task
    )
    assert set(result.forecasts["forecast_policy"].unique()) == {"recursive"}
    assert set(result.forecasts["target"].unique()) == {"y"}


def test_agreeing_loose_keywords_are_not_treated_as_a_conflict(panel):
    """Only a DISAGREEING keyword is a conflict; a redundant one is merely redundant."""
    idx, bundle = panel
    (task,) = _tasks((2,))
    result = mf.forecasting.run(
        bundle,
        "ols",
        window=_window(idx, 2),
        save_models=False,
        task=task,
        target="y",
        features=task.features,
        horizons=[2],
        forecast_policy="direct",
        target_transform="level",
    )
    _assert_same_forecasts(result, _loose(bundle, idx, horizons=(2,)), "redundant keywords")


# --------------------------------------------------------------------------- #
# Conflict rejection
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "kwargs, named",
    [
        (dict(target="x0"), "target"),
        (dict(features=None), None),  # None means "not supplied", so this must NOT raise
        (dict(target_transform="growth"), "target_transform"),
        (dict(forecast_policy="recursive"), "forecast_policy"),
        (dict(horizon=5), "horizon"),
        (dict(horizons=[1, 4]), "horizons"),
    ],
)
def test_a_disagreeing_loose_keyword_is_refused(panel, kwargs, named):
    idx, bundle = panel
    (task,) = _tasks((2,))
    call = dict(window=_window(idx, 5), save_models=False, task=task)
    if named is None:
        mf.forecasting.run(bundle, "ols", **call, **kwargs)
        return
    with pytest.raises(ValueError) as exc:
        mf.forecasting.run(bundle, "ols", **call, **kwargs)
    message = str(exc.value)
    assert named in message, "the refusal must name the keyword the caller passed"
    assert "one or the other" in message, "and say what to do instead"


def test_a_disagreeing_features_spec_is_refused(panel):
    idx, bundle = panel
    (task,) = _tasks((2,))
    with pytest.raises(ValueError, match="features"):
        mf.forecasting.run(
            bundle,
            "ols",
            window=_window(idx, 2),
            save_models=False,
            task=task,
            features=_features("x0"),
        )


def test_a_task_for_another_model_is_refused(panel):
    """A task naming a different model than the one being fitted is the exact
    divergence the task exists to prevent: cache identity and fit would describe
    different arms."""
    idx, bundle = panel
    (task,) = _tasks((1,))
    with pytest.raises(ValueError, match="model"):
        mf.forecasting.run(
            bundle, "ridge", window=_window(idx, 1), save_models=False, task=task
        )


def test_task_and_tasks_together_are_refused(panel):
    idx, bundle = panel
    tasks = _tasks((1, 2))
    with pytest.raises(ValueError, match="not both"):
        mf.forecasting.run(
            bundle,
            "ols",
            window=_window(idx, 2),
            save_models=False,
            task=tasks[0],
            tasks=tasks,
        )


def test_a_mixed_task_sequence_is_refused_at_the_runner(panel):
    idx, bundle = panel
    tasks = _tasks((1,)) + _tasks((2,), policy="recursive")
    with pytest.raises(ValueError, match="ONE cell"):
        mf.forecasting.run(
            bundle, "ols", window=_window(idx, 2), save_models=False, tasks=tasks
        )


def test_a_duplicate_horizon_is_refused_exactly_as_the_loose_path_refuses_it(panel):
    """The task path must not invent its own answer to a question the runner
    already answers."""
    idx, bundle = panel
    tasks = _tasks((1,)) + _tasks((1,))
    with pytest.raises(ValueError, match="unique"):
        mf.forecasting.run(
            bundle, "ols", window=_window(idx, 1), save_models=False, tasks=tasks
        )


def test_an_empty_task_sequence_is_refused(panel):
    idx, bundle = panel
    with pytest.raises(ValueError, match="at least one task"):
        mf.forecasting.run(
            bundle, "ols", window=_window(idx, 1), save_models=False, tasks=[]
        )


# --------------------------------------------------------------------------- #
# The public surface is untouched
# --------------------------------------------------------------------------- #

def test_the_loose_keyword_surface_is_unchanged_when_no_task_is_passed(panel):
    """`forecasting.run` is public and documented with loose keywords.

    The task argument is additive: a call that passes none of it must behave exactly as
    it did, including `run_forecast`'s alias spelling.
    """
    idx, bundle = panel
    loose = _loose(bundle, idx, horizons=(1, 2))
    alias = mf.forecasting.run_forecast(
        bundle,
        "ols",
        window=_window(idx, 2),
        features=_features("y"),
        target="y",
        horizons=[1, 2],
        target_transform="level",
        save_models=False,
    )
    _assert_same_forecasts(loose, alias, "run vs run_forecast alias")


def test_run_stays_atomic_under_a_task(panel):
    """One model per call is a contract of `run`, not of the keyword spelling."""
    idx, bundle = panel
    (task,) = _tasks((1,))
    with pytest.raises(TypeError, match="ONE model"):
        mf.forecasting.run(
            bundle,
            ["ols", "ridge"],
            window=_window(idx, 1),
            save_models=False,
            task=dc.replace(task, model=["ols", "ridge"]),
        )
