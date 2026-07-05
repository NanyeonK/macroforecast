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
sweep) -- design B's under-coverage is stable across all of those.

WP-A1 Step 0 resolution: design B's ~0.82 rate (vs the naive 1-alpha=0.90
target) was cross-checked against R's own ``MCS::MCSprocedure`` on the
identical design (subprocess-Rscript bridge, R=200 replications,
n_boot=500) and R independently reproduces the SAME ~0.82 coverage
(rate=0.8200, CI99=[0.7401,0.8841]). Since the reference R implementation
shows the identical behavior, this is a genuine property of the
Hansen-Lunde-Nason MCS procedure at an exact global null with many tied
models, not a macroforecast bug -- no code change was made. Design B's test
is now a documented-behavior regression band (see
``_EQUAL_SET_DOCUMENTED_BEHAVIOR_NOTE`` in this file and the
``model_confidence_set`` entry in ``docs/reference/tests.md``), not an
``xfail``. Design A (dominant-model coverage) is unaffected and remains a
straightforward ``>=1-alpha`` floor check.
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


_EQUAL_SET_DOCUMENTED_BEHAVIOR_NOTE = (
    "WP-V3 found P(all 5 tied models jointly retained) ~= 0.82 against a "
    "naive 1-alpha=0.90 target (alpha=.10), confirmed stable across a wide "
    "diagnostic sweep (n_boot in {1000,2000}, n_origins in {100,200}, "
    "with/without the shared common factor -- see v3_mc_results.md finding "
    "3). WP-A1 Step 0 then independently cross-checked this EXACT design "
    "(same K=5/n=100/idio_sd=0.3/alpha=.10/block_length=5 DGP and params) "
    "against R's OWN canonical MCS::MCSprocedure via the subprocess-Rscript "
    "bridge (R=200 replications, n_boot=500, MCSprocedure(..., "
    "statistic='Tmax', k=5, min.k=3)): R gives rate=0.8200, CI99=[0.7401, "
    "0.8841] -- essentially identical to this suite's own longer-run (n_reps"
    "=1000) measurement of rate=0.8180, CI99=[0.7846,0.8483]. Since R's own "
    "reference implementation shows the SAME under-coverage on the SAME "
    "design, this is a genuine property of the Hansen-Lunde-Nason MCS "
    "sequential-elimination procedure under an exact global null with many "
    "tied models -- not a macroforecast bug -- so no code change was made. "
    "This is now a DOCUMENTED-BEHAVIOR regression band (not a >=1-alpha "
    "coverage-floor assertion): see the model_confidence_set entry in "
    "docs/reference/tests.md for the interpretation note (what alpha means "
    "under a global null with many ties). A future rate drifting up toward "
    "~0.90 (an unexpected fix upstream) or down below ~0.70 (a new "
    "regression) should fail this band loudly and be re-investigated, not "
    "silently absorbed by widening the band further."
)


@pytest.mark.mc
@pytest.mark.timeout(300)
def test_mcs_equal_losers_global_null_coverage_documented_behavior() -> None:
    """Design B: all K=5 models tied. P(all 5 retained) is ~0.82 here, NOT the
    naive 1-alpha=0.90 target -- see ``_EQUAL_SET_DOCUMENTED_BEHAVIOR_NOTE``:
    this is a confirmed property of the MCS procedure itself under a global
    null with many ties (matched independently by R's own MCS::MCSprocedure,
    WP-A1 Step 0), not a macroforecast defect, so this checks a documented-
    behavior band around the known ~0.82 rate rather than the 1-alpha floor.
    """

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
    rate = n_ok / n_reps
    # Documented-behavior band (WP-A1, see _EQUAL_SET_DOCUMENTED_BEHAVIOR_NOTE):
    # NOT a 1-alpha coverage-floor check. Centered on the R-cross-validated
    # ~0.82 empirical property of the MCS procedure at this exact global-null,
    # many-ties design.
    documented_lo, documented_hi = 0.70, 0.90
    in_band = documented_lo <= rate < documented_hi
    record(
        test="model_confidence_set",
        design=(
            f"K=5, all models tied (idio_sd={idio_sd}, n={n}); DOCUMENTED "
            "BEHAVIOR band (not a 1-alpha coverage floor -- see WP-A1 Step 0)"
        ),
        nominal_alpha=alpha,
        n_reps=n_reps,
        n_rejections=n_ok,
        verdict="PASS (matches documented ~0.82 MCS behavior)" if in_band else "UNEXPECTED -- investigate",
        note=(
            "R MCS::MCSprocedure cross-check (WP-A1 Step 0): rate=0.8200 "
            "CI99=[0.7401,0.8841], R=200, n_boot=500 -- matches this Python "
            "measurement; interpretation note in docs/reference/tests.md"
        ),
        extra={"documented_band": [documented_lo, documented_hi]},
    )
    assert in_band, (
        f"model_confidence_set equal-losers global-null rate moved outside "
        f"the documented ~0.82 behavior band: rate={rate:.4f} "
        f"CI99=[{lo:.4f},{hi:.4f}] expected in [{documented_lo},{documented_hi}) "
        "-- re-investigate rather than widen this band (see "
        "_EQUAL_SET_DOCUMENTED_BEHAVIOR_NOTE)."
    )
