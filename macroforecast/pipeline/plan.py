"""Compile an arm's overrides once, so every site reads the same answer.

An ``Arm`` may override the spec-level ``window`` and ``preprocessing`` (and, with
the latter, its ``preprocessing_policy``). The rule is small -- "the arm's value if
it set one, otherwise the spec's" -- and that is exactly why it had been written out
inline at four separate sites, governing three different things:

===========================  =========================================================
site                         what its answer decides
===========================  =========================================================
``run.py`` (run() kwargs)    what is actually executed
``run.py`` (store namespace) which cached preprocessor fits are considered reusable
``result_store.py``          what the result digest records the run as
===========================  =========================================================

Nothing forced the copies to agree, and the two ``window`` copies did not even use
the same predicate (``getattr(arm, "window", None) is not None`` in one,
``arm.window is not None`` in the other). A disagreement would be silent in the
worst way: the run is one thing and the digest says another, or a cached fit made
under one policy is served to a run under a different one.

This module states the rule once. The consumers ask it rather than restating it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CompiledPreprocessPlan:
    """Which preprocessing runs for one arm, and under which policy.

    ``policy_input`` is deliberately the RAW value (a string, ``None``, or a
    ``StagePolicy``) rather than the resolved object, because that is what
    ``run()`` takes; resolution happens in :meth:`resolve_policy` for the callers
    that need a concrete ``StagePolicy`` -- namely the store namespace, which must
    distinguish two ``fixed_reference`` policies that differ only in their bounds.
    """

    spec: Any | None
    policy_input: Any | None
    source: str
    """``"arm"`` if the arm overrode preprocessing, else ``"spec"``."""

    def resolve_policy(self) -> Any | None:
        """The concrete ``StagePolicy`` this arm's preprocessing is fit under.

        ``None`` when there is no preprocessing at all -- there is no policy to
        speak of, and the store is not namespaced by one.
        """
        if self.spec is None:
            return None
        from macroforecast.meta import get_config
        from macroforecast.window.policy import resolve_stage_policy

        default_scope = str(get_config()["default_preprocessing_scope"])
        return resolve_stage_policy(self.policy_input, default_scope=default_scope)


@dataclass(frozen=True)
class CompiledArmPlan:
    """One arm's resolved overrides, compiled once per (spec, arm)."""

    window: Any | None
    window_source: str
    preprocess: CompiledPreprocessPlan


def compile_arm_plan(spec: Any, arm: Any) -> CompiledArmPlan:
    """Resolve ``arm``'s overrides against ``spec``.

    Structurally typed on purpose: this reads ``.window``, ``.preprocessing`` and
    ``.preprocessing_policy`` off both objects and imports neither, so the pipeline
    spec module does not have to import this one back.

    ``getattr`` rather than plain attribute access for ``window``, because that is
    the more permissive of the two predicates that were in use, and narrowing it
    here would be a behaviour change smuggled in under a refactor.
    """
    arm_window = getattr(arm, "window", None)
    if arm_window is not None:
        window, window_source = arm_window, "arm"
    else:
        window, window_source = getattr(spec, "window", None), "spec"

    arm_preprocessing = getattr(arm, "preprocessing", None)
    if arm_preprocessing is not None:
        # An arm that overrides preprocessing also owns its policy. The spec-level
        # policy must NOT leak in: it would namespace the preprocessor store by a
        # policy that is not the one being fit under.
        preprocess = CompiledPreprocessPlan(
            spec=arm_preprocessing,
            policy_input=getattr(arm, "preprocessing_policy", None),
            source="arm",
        )
    else:
        preprocess = CompiledPreprocessPlan(
            spec=getattr(spec, "preprocessing", None),
            policy_input=getattr(spec, "preprocessing_policy", None),
            source="spec",
        )

    return CompiledArmPlan(window=window, window_source=window_source, preprocess=preprocess)


__all__ = ["CompiledArmPlan", "CompiledPreprocessPlan", "compile_arm_plan"]
