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

What is NOT unified here: `forecasting/policy_config.py` still re-derives transform,
horizon and target mode per policy when `forecasting.run()` is entered directly. That
is the larger half of A2 and needs the runner to accept a task object rather than
loose keywords.
"""
from __future__ import annotations

import dataclasses as _dc
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


def resolve_forecast_task(
    spec: Any,
    arm: Any,
    target: Any,
    horizon: int,
) -> ResolvedForecastTask:
    """Resolve one cell's task, once.

    Every consumer -- execution, result-store identity, checkpoint identity,
    provenance -- should call this rather than repeating the policy override and the
    feature retarget.
    """
    resolved_target = effective_target(spec, arm, target)
    return ResolvedForecastTask(
        target=resolved_target,
        horizon=int(horizon),
        features=retarget_features(arm.features, target.name, arm_name=arm.name),
        arm_name=arm.name,
        model=arm.model,
    )
