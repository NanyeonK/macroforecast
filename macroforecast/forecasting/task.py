"""One resolver for one forecast task.

Before this, resolving "which task is this cell" happened in two places that had
drifted:

  pipeline/run.py            retargets the arm's FeatureSpec, RAISING on failure
  pipeline/result_store.py   retargets it again, SWALLOWING the failure and
                             returning the un-retargeted spec

The execution path and the cache-identity path therefore disagreed about what a
failed retarget means. In practice execution raises first so the digest is never
used, but "in practice the wrong one never runs" is not a contract -- and it is
exactly the kind of divergence a single resolver removes by construction.

`_effective_target_for_arm` (the policy override) was also called from three sites.
Both identity call sites already passed the overridden target, so there was no cache
bug to fix -- verified before this change, not assumed.

The runner now accepts the task object itself (`run(..., task=...)` / `tasks=...`), so
the pipeline resolves a cell ONCE and hands the same object to execution, checkpoint
identity, result-store identity and the provenance echo. The loose keywords remain the
public surface and are still accepted alone; supplying both is refused rather than
silently reconciled (see `runner.run`).

What is NOT unified here: `forecasting/policy_config.py` still re-derives transform,
horizon and target mode per policy from the resolved keywords. That is a separate step
-- the runner normalizes a task back into those keywords at its entry, which is what
makes this refactor behavior-preserving.
"""
from __future__ import annotations

import dataclasses as _dc
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

# No import of pipeline types here, not even under TYPE_CHECKING. ``forecasting`` sits
# BELOW ``pipeline``, and the architecture guard rejects the upward reference -- it
# caught this exact line during A2 and it was right to. A forecast task is a statement
# about one forecast and must be expressible without the study-orchestration types.
#
# The parameters below are therefore structural: anything carrying ``.name`` /
# ``.policy`` / ``.transform`` for a target, ``.name`` / ``.features`` / ``.model`` for
# an arm, and ``.policy_overrides`` for the spec. That is also what lets a direct
# ``forecasting.run()`` reach the same resolver the pipeline uses.


class FeatureRetargetError(ValueError):
    """An arm's feature spec could not be aligned to the target being forecast.

    Raised rather than silently returning the original spec: a feature spec still
    pointing at another target would produce forecasts for the wrong series, and a
    cache identity computed from it would describe a task that was never run.
    """


@dataclass(frozen=True)
class ResolvedForecastTask:
    """Everything that identifies one (target, horizon, arm) forecast.

    Built once by :func:`resolve_forecast_task` and consumed by execution, cache
    identity, checkpoint identity and the provenance echo, so those cannot disagree
    about which question is being answered.
    """

    target: Any  # structural: carries .name / .policy / .transform
    horizon: int
    features: Any
    arm_name: str
    model: Any

    @property
    def target_name(self) -> str:
        return self.target.name

    @property
    def forecast_policy(self) -> str:
        return self.target.policy

    @property
    def target_transform(self) -> str:
        return self.target.transform


def effective_target(
    spec: Any, arm: Any, target: Any
) -> Any:
    """Apply this spec's per-(arm, target) forecast-policy override, if any."""
    policy = spec.policy_overrides.get((arm.name, target.name))
    if policy is None or policy == target.policy:
        return target
    return _dc.replace(target, policy=policy)


def retarget_features(features: Any, target_name: str, *, arm_name: str = "") -> Any:
    """Align an arm's feature spec to *target_name*.

    A multi-target pipeline runs every arm for every target, but a feature spec
    carries a single target, so it has to be re-pointed per cell.
    """
    if features is None:
        return None
    needs = (
        getattr(features, "target", None) != target_name
        or bool(getattr(features, "targets", ()))
    )
    if not needs:
        return features
    kwargs: dict[str, Any] = {"target": target_name}
    if getattr(features, "targets", None):
        kwargs["targets"] = ()
    try:
        return _dc.replace(features, **kwargs)
    except Exception as exc:
        where = f" of arm {arm_name!r}" if arm_name else ""
        raise FeatureRetargetError(
            f"could not re-target feature spec{where} to {target_name!r}: {exc}"
        ) from exc


def resolve_forecast_tasks(
    spec: Any,
    arm: Any,
    target: Any,
    horizons: Iterable[int],
) -> tuple[ResolvedForecastTask, ...]:
    """Resolve one (target, arm) cell's tasks, resolving the design exactly ONCE.

    The policy override and the feature retarget are horizon-INDEPENDENT: they answer
    "which series, under which policy, from which features", and a horizon-group cell
    asks that question once and then forecasts several steps ahead. So they are
    computed once here and only the horizon is stamped per task -- which is what makes
    "resolve once" a structural property rather than a convention each caller has to
    remember. Every task shares one ``features`` OBJECT, exactly as the multi-horizon
    runner already shared one ``features`` argument across its per-horizon calls.

    Every consumer -- execution, result-store identity, checkpoint identity,
    provenance -- should take its answer from these tasks rather than repeating the
    resolution.

    Raises :class:`FeatureRetargetError` when the arm's feature spec cannot be aligned
    to *target*, before any horizon is stamped: a cell that cannot be resolved has no
    tasks at all, rather than some horizons resolved and some not.
    """
    values = tuple(int(horizon) for horizon in horizons)
    if not values:
        raise ValueError("a forecast cell must request at least one horizon")
    resolved_target = effective_target(spec, arm, target)
    # Retargeting is keyed on the target's NAME, and the policy override replaces only
    # ``policy``, so ``target`` and ``resolved_target`` always name the same series.
    features = retarget_features(arm.features, target.name, arm_name=arm.name)
    return tuple(
        ResolvedForecastTask(
            target=resolved_target,
            horizon=horizon,
            features=features,
            arm_name=arm.name,
            model=arm.model,
        )
        for horizon in values
    )


def resolve_forecast_task(
    spec: Any,
    arm: Any,
    target: Any,
    horizon: int,
) -> ResolvedForecastTask:
    """Resolve one single-horizon cell's task, once.

    A thin single-horizon spelling of :func:`resolve_forecast_tasks` -- one code path,
    so the two cannot drift.
    """
    return resolve_forecast_tasks(spec, arm, target, (horizon,))[0]


def _same(left: Any, right: Any) -> bool:
    """Structural equality that never raises on an exotic user object."""
    if left is right:
        return True
    try:
        return bool(left == right)
    except Exception:
        return False


# The fields a task sequence describes ONE of: a sequence spanning two of any of them
# is not a cell, and the horizon-group execution path would quietly forecast the first
# task's design for all of them. ``horizon`` is deliberately absent -- it is the one
# dimension a sequence exists to vary.
#
# The target is compared through its derived scalars rather than as an OBJECT: two
# equivalent targets that are not ``==`` (a structural stand-in without dataclass
# equality) describe the same forecast, and refusing them would reject a correct caller
# for a reason unrelated to forecasting.
_SHARED_TASK_FIELDS = (
    "target_name",
    "forecast_policy",
    "target_transform",
    "arm_name",
    "model",
    "features",
)


def validate_task_sequence(
    tasks: Sequence[ResolvedForecastTask],
) -> tuple[ResolvedForecastTask, tuple[int, ...]]:
    """Return ``(shared_task, horizons)`` for a sequence describing one cell.

    Refuses a MIXED sequence -- two targets, arms, models, forecast policies,
    transforms or feature specs -- because the consumer of a sequence (the grouped
    multi-horizon runner call) applies one design to every horizon, so a mixed
    sequence would silently forecast the first task's design under the others' labels.

    Every requested horizon is retained, in the order given: this function must never
    collapse a horizon group. Positivity and uniqueness are left to the runner's own
    horizon validation so the task and loose-keyword paths refuse identically.
    """
    items = tuple(tasks)
    if not items:
        raise ValueError("a forecast task sequence must contain at least one task")
    shared = items[0]
    for other in items[1:]:
        for field_name in _SHARED_TASK_FIELDS:
            if not _same(getattr(shared, field_name), getattr(other, field_name)):
                raise ValueError(
                    "a forecast task sequence must describe ONE cell; got tasks "
                    f"disagreeing on {field_name!r} "
                    f"(h{shared.horizon} vs h{other.horizon}). Run one sequence per "
                    "(target, arm) cell."
                )
    return shared, tuple(task.horizon for task in items)


__all__ = [
    "FeatureRetargetError",
    "ResolvedForecastTask",
    "effective_target",
    "resolve_forecast_task",
    "resolve_forecast_tasks",
    "retarget_features",
    "validate_task_sequence",
]
