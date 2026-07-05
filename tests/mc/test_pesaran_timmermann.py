"""WP-V3 target #4: ``pesaran_timmermann_test`` size under independent directions.

DGP: ``y_true`` and ``y_pred`` are drawn independently from N(0,1) -- the
predicted sign carries zero information about the true sign, so the true
directional "success ratio" is exactly the chance rate implied by each
series' own marginal sign distribution (both symmetric here, chance=0.5).
Under this null, ``pesaran_timmermann_test``'s one-sided upper-tail test
should reject at rate ~= alpha.
"""
from __future__ import annotations

import numpy as np
import pytest

import macroforecast as mf

from tests.mc.conftest import clopper_pearson, record, spawn_generators

N_REPS = 5000
ALPHAS = (0.05, 0.10)


_N50_DISTORTION_REASON = (
    "CONFIRMED at R=15000 (see v3_mc_results.md), not just N_REPS=5000 noise: "
    "rate=0.0559 @alpha=.05 (CI99 [0.0512,0.0609]) and rate=0.1084 @alpha=.10 "
    "(CI99 [0.1020,0.1151]) -- both mildly but genuinely oversized. Diagnosis: "
    "the PT statistic's asymptotic-normal reference (p_hat_var - p_star_var, "
    "Pesaran-Timmermann 1992) is a large-n approximation; n=50 leaves a small "
    "residual size distortion consistent with the known finite-sample behavior "
    "of this asymptotic test (formula itself already parity-checked against "
    "tstests::dac_test / rugarch::DACTest in WP-V0/V1). n=200 in this same "
    "suite is correctly sized on the identical code path, isolating this to "
    "small-n asymptotics rather than a formula/df defect."
)


@pytest.mark.mc
@pytest.mark.timeout(90)
@pytest.mark.parametrize(
    "n",
    [
        pytest.param(50, marks=pytest.mark.xfail(reason=_N50_DISTORTION_REASON, strict=True)),
        200,
    ],
)
def test_pesaran_timmermann_size_independent_directions(n: int) -> None:
    gens = spawn_generators(N_REPS, salt=4_000_000 + n)
    pvals = np.empty(N_REPS)
    for i, rng in enumerate(gens):
        y_true = rng.normal(size=n)
        y_pred = rng.normal(size=n)
        result = mf.tests.pesaran_timmermann_test(y_true, y_pred)
        pvals[i] = result.p_value

    violations = []
    for alpha in ALPHAS:
        n_reject = int(np.sum(pvals < alpha))
        lo, hi = clopper_pearson(n_reject, N_REPS, conf=0.99)
        in_band = lo <= alpha <= hi
        verdict = "PASS (in band)" if in_band else ("OVERSIZED" if alpha < lo else "UNDERSIZED")
        record(
            test="pesaran_timmermann_test",
            design=f"independent directions, n={n}",
            nominal_alpha=alpha,
            n_reps=N_REPS,
            n_rejections=n_reject,
            verdict=verdict,
            extra={"n": n},
        )
        if not in_band:
            violations.append(
                f"alpha={alpha} empirical_rate={n_reject / N_REPS:.4f} CI99=[{lo:.4f},{hi:.4f}] ({verdict})"
            )
    assert not violations, f"pesaran_timmermann_test size distortion at n={n}: " + "; ".join(violations)
