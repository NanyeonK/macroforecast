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

import os
import random
import warnings
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


def apply_reproducibility_mode(
    *,
    mode: str,
    seed: int,
    configure_torch: bool = True,
) -> dict:
    """Install global RNG state + determinism flags for the given mode.

    Called once at the start of each :func:`execute_recipe` invocation so
    that libraries downstream of our explicit ``resolve_seed`` calls (numpy
    globals, torch, cudnn, scikit-learn defaults, etc.) share the same
    deterministic regime.

    Behaviour per mode:

    * ``strict_reproducible`` — pin Python/NumPy/torch seeds AND enable
      torch deterministic algorithms + cuDNN ``deterministic=True`` +
      ``benchmark=False``. Set ``CUBLAS_WORKSPACE_CONFIG=':4096:8'``
      (required by torch for CUDA determinism if CUDA is active) when not
      already set. Emit a ``RuntimeWarning`` if ``PYTHONHASHSEED`` is not
      set — that one MUST be configured in the shell before Python starts.

    * ``seeded_reproducible`` — pin Python/NumPy/torch seeds but do not
      flip cuDNN / deterministic-algorithms flags (small numerical drift
      across library versions is accepted).

    * ``best_effort`` — identical to ``seeded_reproducible`` at install
      time; the label exists to mark runs that callers explicitly do not
      want counted as strict for CI regression checks.

    * ``exploratory`` — **no-op**. Whatever the caller's pre-existing RNG
      state is stays untouched. The variant will still receive a fresh,
      non-deterministic seed from :func:`current_seed` but we do not
      reset global state.

    Args:
        mode: One of ``VALID_MODES``.
        seed: Base seed to install (typically the value returned by
            :func:`resolve_seed` for the recipe root).
        configure_torch: Set False to skip torch/cuDNN configuration even
            when torch is importable. Useful for unit tests that want to
            assert the NumPy/Python side independently.

    Returns:
        A dict summarising what was installed — suitable for embedding in
        the run manifest. Keys: ``mode``, ``python_hash_seed``,
        ``numpy_seed_set``, ``torch_seed_set``, ``cudnn_deterministic``,
        ``cudnn_benchmark``, ``torch_deterministic_algorithms``,
        ``cublas_workspace_config``.

    Raises:
        ValueError: if ``mode`` is not in :data:`VALID_MODES`.
    """

    if mode not in VALID_MODES:
        raise ValueError(f"unknown reproducibility_mode: {mode!r}")

    summary: dict = {
        "mode": mode,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "numpy_seed_set": False,
        "torch_seed_set": False,
        "cudnn_deterministic": None,
        "cudnn_benchmark": None,
        "torch_deterministic_algorithms": None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }

    if mode == "exploratory":
        return summary

    # Python + NumPy global RNG state — applied for all three non-exploratory modes.
    random.seed(seed)
    np.random.seed(seed)
    summary["numpy_seed_set"] = True

    if configure_torch:
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None:
            torch.manual_seed(seed)
            try:
                torch.cuda.manual_seed_all(seed)
            except Exception:  # CUDA unavailable or not compiled in
                pass
            summary["torch_seed_set"] = True

            if mode == "strict_reproducible":
                try:
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                    summary["cudnn_deterministic"] = True
                    summary["cudnn_benchmark"] = False
                except AttributeError:
                    pass
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                    summary["torch_deterministic_algorithms"] = True
                except (AttributeError, RuntimeError):
                    pass

    if mode == "strict_reproducible":
        # CUBLAS workspace is required by torch for CUDA deterministic algorithms.
        # Only set it if not already configured so we do not override a user's explicit choice.
        if not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        summary["cublas_workspace_config"] = os.environ["CUBLAS_WORKSPACE_CONFIG"]

        if summary["python_hash_seed"] is None:
            warnings.warn(
                "reproducibility_mode='strict_reproducible' but PYTHONHASHSEED "
                "is not set. Hash-randomised structures (sets, dicts, "
                "PYTHONHASHSEED-dependent paths) may still vary across runs. "
                "Set PYTHONHASHSEED=0 in the shell before launching Python to "
                "fully pin hash ordering.",
                RuntimeWarning,
                stacklevel=2,
            )

    return summary
