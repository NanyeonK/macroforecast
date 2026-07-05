"""WP-V3 target #6: ``giacomini_white_test`` size under equal conditional predictive ability.

DGP: same MA(h-1) equal-accuracy forecast-error construction as the
``dm_test`` size suite (independent innovations, identical variance per
model) -- H0 (conditional AND unconditional equal predictive ability) is
true by construction: the loss differential ``dL_t`` has zero mean and no
functional dependence on its own lag (the default GW instrument, ``[1,
dL_{t-h}]``), since the two loss series are built from independent
innovations. The GW statistic is a chi2(q) Wald test (q=2 with the default
instrument set), not a t-test, so it is checked against its own nominal
alpha independently of the ``dm_test`` size suite (different reference
distribution, different HAC construction: Newey-West with ``h-1`` lags and
Bartlett weights, vs. dm_test's unweighted acf-kernel sum).
"""
from __future__ import annotations

import numpy as np
import pytest

import macroforecast as mf

from tests.mc.conftest import clopper_pearson, record, spawn_generators

N_REPS = 5000
ALPHAS = (0.05, 0.10)


def _simulate_ma_errors(rng: np.random.Generator, n: int, h: int) -> np.ndarray:
    eps = rng.normal(size=n + h - 1)
    return np.convolve(eps, np.ones(h), mode="valid")


_H4_DISTORTION_REASON = (
    "CONFIRMED, not sample-size noise: rate stays ~1.5-1.8x nominal alpha at "
    "n=50, 200, 500, 1000, 2000, AND 5000 (see v3_mc_results.md) -- e.g. at "
    "n=5000 mean(statistic)=2.29 vs the chi2(2) mean of 2.00, rate@alpha=.05 "
    "=0.072. Diagnosis: a follow-up sweep over h=1..4 at fixed n=500 shows a "
    "MONOTONIC relationship between h (i.e. the number of HAC lags, h-1, in "
    "the Newey-West/Bartlett covariance) and the size distortion: h=1 (zero "
    "HAC lags) is correctly calibrated (mean stat 1.95, rate 0.045); h=2,3,4 "
    "(1, 2, 3 HAC lags) show increasing over-rejection (mean stat 2.27, 2.45, "
    "2.57; rate 0.070, 0.079, 0.084 at n=500). This is consistent with the "
    "well-documented finite-sample behavior of HAC-robust Wald tests, where "
    "more HAC lags relative to the effective sample size inflate the "
    "estimated-covariance-based Wald statistic (same qualitative mechanism "
    "as the dm_test HLN small-sample correction exists to address -- but "
    "giacomini_white_test has no analogous small-sample correction here). "
    "Shrinks only very slowly with n (does not resolve by n=5000), so this "
    "is reported as a genuine, still-open distortion rather than dismissed "
    "as small-n noise."
)


@pytest.mark.mc
@pytest.mark.timeout(90)
@pytest.mark.parametrize("n", [50, 200])
@pytest.mark.parametrize(
    "h",
    [
        1,
        pytest.param(4, marks=pytest.mark.xfail(reason=_H4_DISTORTION_REASON, strict=True)),
    ],
)
def test_giacomini_white_size_equal_conditional_accuracy(n: int, h: int) -> None:
    gens = spawn_generators(N_REPS, salt=7_000_000 + n * 10 + h)
    pvals = np.empty(N_REPS)
    for i, rng in enumerate(gens):
        loss_a = _simulate_ma_errors(rng, n, h) ** 2
        loss_b = _simulate_ma_errors(rng, n, h) ** 2
        result = mf.tests.giacomini_white_test(loss_a, loss_b, horizon=h)
        pvals[i] = result.p_value if result.p_value is not None else np.nan

    valid = pvals[~np.isnan(pvals)]
    n_valid = int(valid.size)
    assert n_valid >= 0.95 * N_REPS, (
        f"too many undefined GW statistics: {N_REPS - n_valid}/{N_REPS} (n={n}, h={h})"
    )
    violations = []
    for alpha in ALPHAS:
        n_reject = int(np.sum(valid < alpha))
        lo, hi = clopper_pearson(n_reject, n_valid, conf=0.99)
        in_band = lo <= alpha <= hi
        verdict = "PASS (in band)" if in_band else ("OVERSIZED" if alpha < lo else "UNDERSIZED")
        record(
            test="giacomini_white_test",
            design=f"equal conditional accuracy, MA({h - 1}) errors, n={n}, h={h}",
            nominal_alpha=alpha,
            n_reps=n_valid,
            n_rejections=n_reject,
            verdict=verdict,
            note="default instrument [1, dL_{t-h}]; chi2(df=2) reference",
            extra={"n": n, "h": h},
        )
        if not in_band:
            violations.append(
                f"alpha={alpha} empirical_rate={n_reject / n_valid:.4f} CI99=[{lo:.4f},{hi:.4f}] ({verdict})"
            )
    assert not violations, f"giacomini_white_test size distortion at n={n}, h={h}: " + "; ".join(violations)
