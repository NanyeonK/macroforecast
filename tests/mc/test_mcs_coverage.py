"""WP-V3 target #3: ``model_confidence_set`` (MCS) coverage.

Two designs, both K=5 models, AR(1) common factor + AR(1) idiosyncratic
noise per model (realistic serially-correlated loss panels, not iid):

  A) DOMINANT: one model (``m0``) has a small but real loss advantage, the
     other four are exactly tied (worse). Hansen-Lunde-Nason theory:
     P(true best retained in the MCS) >= 1-alpha asymptotically. The gap is
     tuned deliberately small (delta=0.01 mean loss, n=50 origins) so the
     retention probability sits in an INFORMATIVE range near the 1-alpha
     target rather than saturating at a trivial ~1.0 ceiling (a larger gap
     was tried first and always retained m0 in 300/300 replications --
     uninformative about calibration; see the diagnostic sweep referenced
     below).

  B) EQUAL-LOSERS / GLOBAL NULL: all K=5 models have IDENTICAL expected
     loss. Under complete symmetry every model is tied for "true best", so
     the MCS's own coverage guarantee reduces to: P(all 5 models retained)
     ~= 1-alpha -- this is a clean size-like check of the procedure's
     FIRST-STEP elimination test (the global null of "no model is worse
     than any other"), distinct from design A's single-best-model coverage.

Both designs were checked across n_boot in {1000, 2000}, n_origins in
{50, 100, 200, 500}, alpha in {0.05, 0.10}, with/without the shared common
factor, and multiple fixed block lengths (1/5/10/20/"auto") before writing
this file (see ``.dev-notes/anchor_coverage/v3_mc_results.md`` for the full
sweep) -- design B's under-coverage is stable across all of those, which is
why it is reported as a genuine finding (``xfail(strict=True)``) rather than
tuned away.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf

from tests.mc.conftest import clopper_pearson, record, spawn_generators

N_BOOT = 1000
BLOCK_LENGTH = 5
COMMON_RHO = 0.5
IDIO_RHO = 0.3
COMMON_SD = 0.15


def _make_panel(
    rng: np.random.Generator,
    *,
    base_losses: list[float],
    model_names: list[str],
    n: int,
    idio_sd: float,
) -> pd.DataFrame:
    common = np.empty(n)
    common[0] = rng.normal(0.0, COMMON_SD / np.sqrt(max(1e-9, 1.0 - COMMON_RHO**2)))
    for t in range(1, n):
        common[t] = COMMON_RHO * common[t - 1] + rng.normal(0.0, COMMON_SD)
    rows = []
    for name, base in zip(model_names, base_losses):
        idio = np.empty(n)
        idio[0] = rng.normal(0.0, idio_sd / np.sqrt(max(1e-9, 1.0 - IDIO_RHO**2)))
        for t in range(1, n):
            idio[t] = IDIO_RHO * idio[t - 1] + rng.normal(0.0, idio_sd)
        loss = base + common + idio
        for origin in range(n):
            rows.append(
                {
                    "target": "y",
                    "horizon": 1,
                    "origin": origin,
                    "model_id": name,
                    "squared_error": loss[origin],
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.mc
@pytest.mark.timeout(300)
def test_mcs_retains_true_best_model_near_nominal_coverage() -> None:
    """Design A: a small, real, single-model advantage should be retained ~1-alpha of the time."""

    n, delta, idio_sd, alpha, n_reps = 50, 0.01, 0.4, 0.10, 1000
    model_names = [f"m{i}" for i in range(5)]
    base_losses = [1.0] + [1.0 + delta] * 4
    gens = spawn_generators(n_reps, salt=5_000_001)
    n_ok = 0
    for i, rng in enumerate(gens):
        panel = _make_panel(rng, base_losses=base_losses, model_names=model_names, n=n, idio_sd=idio_sd)
        result = mf.tests.model_confidence_set(
            panel, alpha=alpha, n_boot=N_BOOT, block_length=BLOCK_LENGTH,
            random_state=15_000_000 + i,
        )
        included = set(result["mcs_inclusion"][0]["models"])
        if "m0" in included:
            n_ok += 1

    lo, hi = clopper_pearson(n_ok, n_reps, conf=0.99)
    target = 1.0 - alpha
    # The MCS coverage guarantee is a LOWER bound (P(true best retained) >=
    # 1-alpha), so over-covering (retaining even more often than 1-alpha) is
    # conservative and expected, not a distortion -- only flag if the CI's
    # upper bound falls below the target (a real shortfall below the
    # guaranteed floor).
    meets_floor = hi >= target
    record(
        test="model_confidence_set",
        design=(
            f"K=5, one model with small real advantage (delta={delta}, idio_sd={idio_sd}, "
            f"n={n}); target = P(true best retained) >= 1-alpha"
        ),
        nominal_alpha=alpha,
        n_reps=n_reps,
        n_rejections=n_ok,
        verdict="PASS (meets >=1-alpha coverage floor)" if meets_floor else "COVERAGE DISTORTION (below floor)",
        note=f"target retention rate = 1-alpha = {target:.2f} (one-sided floor check)",
        extra={"target_coverage": target},
    )
    assert meets_floor, (
        f"model_confidence_set dominant-model coverage distortion: rate="
        f"{n_ok / n_reps:.4f} CI99=[{lo:.4f},{hi:.4f}] vs floor={target:.2f}"
    )


_EQUAL_SET_DISTORTION_REASON = (
    "CONFIRMED across a wide diagnostic sweep (see v3_mc_results.md), not a "
    "single-run artifact: P(all 5 tied models jointly retained) came in at "
    "0.82-0.87 against a 1-alpha=0.90 target (alpha=.10) -- and 0.867 against "
    "a 1-alpha=0.95 target (alpha=.05, i.e. an even larger relative shortfall "
    "-- across n_boot in {1000,2000} (rules out bootstrap MC noise), "
    "n_origins in {100,200} (rules out a simple finite-sample effect that "
    "vanishes with more data), and with/without the shared common factor "
    "(rules out cross-model correlation as the sole driver). Diagnosis: this "
    "isolates the distortion to the model_confidence_set / "
    "_iterative_mcs_wide first-step elimination test itself (the exact "
    "_mcs_statistic private helper flagged in WP-V0 as the single biggest "
    "untested gap in tests.py) under-covering the 'no model is worse' global "
    "null by a stable, non-vanishing margin -- not to this MC design's DGP, "
    "block length, or replication budget."
)


@pytest.mark.mc
@pytest.mark.timeout(300)
@pytest.mark.xfail(reason=_EQUAL_SET_DISTORTION_REASON, strict=True)
def test_mcs_equal_losers_global_null_coverage() -> None:
    """Design B: all K=5 models tied -> P(all 5 retained) should be ~1-alpha."""

    n, idio_sd, alpha, n_reps = 100, 0.3, 0.10, 1000
    model_names = [f"m{i}" for i in range(5)]
    base_losses = [1.0] * 5
    gens = spawn_generators(n_reps, salt=5_000_002)
    n_ok = 0
    for i, rng in enumerate(gens):
        panel = _make_panel(rng, base_losses=base_losses, model_names=model_names, n=n, idio_sd=idio_sd)
        result = mf.tests.model_confidence_set(
            panel, alpha=alpha, n_boot=N_BOOT, block_length=BLOCK_LENGTH,
            random_state=16_000_000 + i,
        )
        included = set(result["mcs_inclusion"][0]["models"])
        if len(included) == 5:
            n_ok += 1

    lo, hi = clopper_pearson(n_ok, n_reps, conf=0.99)
    target = 1.0 - alpha
    meets_floor = hi >= target
    record(
        test="model_confidence_set",
        design=f"K=5, all models tied (idio_sd={idio_sd}, n={n}); target = P(all retained) >= 1-alpha",
        nominal_alpha=alpha,
        n_reps=n_reps,
        n_rejections=n_ok,
        verdict="PASS (meets >=1-alpha coverage floor)" if meets_floor else "UNDERCOVERAGE (below floor)",
        note=f"target retention rate = 1-alpha = {target:.2f} (one-sided floor check)",
        extra={"target_coverage": target},
    )
    assert meets_floor, (
        f"model_confidence_set equal-losers global-null undercoverage: rate="
        f"{n_ok / n_reps:.4f} CI99=[{lo:.4f},{hi:.4f}] vs floor={target:.2f}"
    )
