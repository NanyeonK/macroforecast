"""Deterministic per-variant seed resolution.

Replaces hardcoded ``random_state=42`` literals across ``execute_recipe`` and
its downstream callers with a policy that reads a recipe-level
``reproducibility_spec`` and returns concrete integer seeds. Routing every
randomness-consuming call site through ``resolve_seed`` (or the context-aware
``current_seed`` convenience wrapper) guarantees:

* identical recipes produce identical artifacts across machines
* sweep variants derive distinct but deterministic seeds from ``variant_id``
* exploratory runs can opt into non-determinism without code changes

See ``plans/infra/seed_policy.md`` and ``docs/dev/reproducibility_policy.md``
for the operational contract.
"""

from __future__ import annotations

import contextvars
import hashlib
from dataclasses import dataclass

import numpy as np

VALID_MODES = frozenset(
    {
        "strict_reproducible",
        "seeded_reproducible",
        "best_effort",
        "exploratory",
    }
)


@dataclass(frozen=True)
class ReproducibilityContext:
    """Snapshot of the reproducibility inputs for one ``execute_recipe`` call."""

    recipe_id: str
    variant_id: str | None
    reproducibility_spec: dict


_CONTEXT: contextvars.ContextVar[ReproducibilityContext | None] = contextvars.ContextVar(
    "_macrocast_reproducibility_context", default=None
)


def set_context(ctx: ReproducibilityContext) -> contextvars.Token:
    """Install ``ctx`` as the current reproducibility context.

    Returns a token that must be passed to :func:`reset_context` when the
    caller is done (typically in a ``try/finally`` block around the scope
    that sources seeds from the context).
    """

    return _CONTEXT.set(ctx)


def reset_context(token: contextvars.Token) -> None:
    """Restore the reproducibility context that was active before ``set_context``."""

    _CONTEXT.reset(token)


def get_context() -> ReproducibilityContext | None:
    """Return the currently installed reproducibility context (or ``None``)."""

    return _CONTEXT.get()


def resolve_seed(
    *,
    recipe_id: str,
    variant_id: str | None = None,
    reproducibility_spec: dict,
    model_family: str | None = None,
) -> int:
    """Deterministic per-variant seed resolution.

    Modes (selected via ``reproducibility_spec['reproducibility_mode']``):

    * ``strict_reproducible`` — hash-derived per-variant seed; bit-identical
      outputs guaranteed for a fixed ``(recipe_id, variant_id, model_family)``.
    * ``seeded_reproducible`` — single ``base_seed`` for the whole run; the
      default for single-path execution.
    * ``best_effort`` — identical behavior to ``seeded_reproducible``; marks
      runs where BLAS/GPU non-determinism is tolerated.
    * ``exploratory`` — fresh non-deterministic seed per call.

    Raises ``ValueError`` for unknown modes.
    """

    mode = reproducibility_spec.get("reproducibility_mode", "seeded_reproducible")
    if mode not in VALID_MODES:
        raise ValueError(f"unknown reproducibility_mode: {mode!r}")

    base_seed = int(reproducibility_spec.get("seed", 42))

    if mode == "strict_reproducible":
        key = f"{recipe_id}|{variant_id or 'main'}|{model_family or ''}"
        digest = hashlib.sha256(key.encode()).hexdigest()[:8]
        return int(digest, 16) & 0x7FFFFFFF
    if mode == "seeded_reproducible" or mode == "best_effort":
        return base_seed
    if mode == "exploratory":
        return int(np.random.randint(0, 2**31 - 1))
    raise ValueError(f"unreachable reproducibility_mode: {mode!r}")


_DEFAULT_SPEC: dict = {"reproducibility_mode": "seeded_reproducible", "seed": 42}


def current_seed(model_family: str | None = None) -> int:
    """Resolve a seed using the currently-installed reproducibility context.

    Convenience for call sites inside ``execute_recipe`` that only know the
    ``model_family`` string. When no context is active (direct invocation
    outside ``execute_recipe``), falls back to ``seed=42`` under
    ``seeded_reproducible`` mode — matching the pre-Phase-0 behavior.
    """

    ctx = _CONTEXT.get()
    if ctx is None:
        return resolve_seed(
            recipe_id="",
            variant_id=None,
            reproducibility_spec=_DEFAULT_SPEC,
            model_family=model_family,
        )
    return resolve_seed(
        recipe_id=ctx.recipe_id,
        variant_id=ctx.variant_id,
        reproducibility_spec=ctx.reproducibility_spec,
        model_family=model_family,
    )
