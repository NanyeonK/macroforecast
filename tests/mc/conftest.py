"""Shared fixtures/helpers for ``tests/mc/`` (WP-V3 Monte Carlo size/power harness).

Purpose (see also ``.dev-notes/anchor_coverage/v3_mc_results.md``): the
parity harnesses (V1 R-parity, V2 model anchors) prove macroforecast MATCHES
a reference implementation on fixed fixtures; this harness proves the
STATISTICAL BEHAVIOR of the inference-critical forecast-comparison tests is
correct. A test whose rejection rate under the null is far from its nominal
alpha is broken for referees even when it matches a reference formula
bit-for-bit -- this catches distribution/HAC/df errors that a shared-formula
parity check can hide by construction (both sides share the bug).

Every MC design in this directory:
  - fixes a master seed (``MC_MASTER_SEED``) and spawns one independent
    ``np.random.Generator`` per replication via
    ``numpy.random.SeedSequence(MC_MASTER_SEED, spawn_key=(salt,)).spawn(n)``
    -- no shared mutable RNG state across replications, no stream reuse
    across design cells (each cell uses its own integer ``salt``);
  - reports a Clopper-Pearson exact-binomial confidence interval around the
    empirical rejection rate at a FIXED 99% confidence level and compares it
    to the nominal alpha -- the band is fixed before looking at results and
    is never widened after the fact to force a PASS (that is exactly the
    failure mode this WP exists to prevent -- a distortion is a finding,
    not a reason to loosen the check);
  - is marked ``@pytest.mark.mc`` (excluded from default/CI runs -- these
    replications are 10-1000x more expensive than a formula/parity check)
    and carries a per-test ``@pytest.mark.timeout``.

Run with: ``pytest tests/mc/ -m mc -s`` (``-s`` to see the printed per-cell
diagnostics as they run). Results are also machine-written to
``_mc_raw_results.json`` in this directory (session-scoped, overwritten each
run) -- that file is the source for ``v3_mc_results.md``, not hand-transcribed
terminal output.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import stats as sp_stats

# Fixed master seed for the whole WP-V3 harness. Every design cell spawns
# independent per-replication seeds from this via SeedSequence.spawn(), so
# no replication's draw depends on how many replications ran before it, and
# distinct cells (distinct ``salt``) never share a stream.
MC_MASTER_SEED = 20260705

_RESULTS_PATH = Path(__file__).parent / "_mc_raw_results.json"
_RESULTS: list[dict[str, Any]] = []


def spawn_generators(n: int, *, salt: int) -> list[np.random.Generator]:
    """Return ``n`` independent Generators, reproducible given (MASTER_SEED, salt).

    ``salt`` must be unique per design cell (test x parameter combination)
    so two cells never draw the same stream even though both derive from
    the same master seed.
    """

    ss = np.random.SeedSequence(MC_MASTER_SEED, spawn_key=(int(salt),))
    return [np.random.default_rng(child) for child in ss.spawn(n)]


def clopper_pearson(k: int, n: int, *, conf: float = 0.99) -> tuple[float, float]:
    """Exact binomial confidence interval for ``k`` successes out of ``n`` trials."""

    alpha = 1.0 - conf
    lo = 0.0 if k == 0 else float(sp_stats.beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(sp_stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


def record(
    *,
    test: str,
    design: str,
    nominal_alpha: float,
    n_reps: int,
    n_rejections: int,
    verdict: str,
    note: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one design-cell result to the session log; print it (visible with ``-s``)."""

    rate = n_rejections / n_reps
    lo, hi = clopper_pearson(n_rejections, n_reps, conf=0.99)
    row: dict[str, Any] = {
        "test": test,
        "design": design,
        "nominal_alpha": nominal_alpha,
        "n_reps": n_reps,
        "n_rejections": n_rejections,
        "empirical_rate": rate,
        "ci99_lo": lo,
        "ci99_hi": hi,
        "band_width_pct": (hi - lo) * 100.0,
        "verdict": verdict,
        "note": note,
    }
    if extra:
        row.update(extra)
    _RESULTS.append(row)
    print(
        f"[mc] {test} | {design} | alpha={nominal_alpha} | R={n_reps} | "
        f"reject={n_rejections} ({rate:.4f}) | CI99=[{lo:.4f},{hi:.4f}] | {verdict}"
    )
    return row


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:  # noqa: ARG001
    if not _RESULTS:
        return
    payload = {"master_seed": MC_MASTER_SEED, "results": _RESULTS}
    _RESULTS_PATH.write_text(json.dumps(payload, indent=2, default=float) + "\n")
