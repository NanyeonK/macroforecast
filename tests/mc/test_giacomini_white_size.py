"""WP-V3/WP-A1 target #6: ``giacomini_white_test`` size under equal conditional
predictive ability.

DGP: same MA(h-1) equal-accuracy forecast-error construction as the
``dm_test`` size suite (independent innovations, identical variance per
model) -- H0 (conditional AND unconditional equal predictive ability) is
true by construction: the loss differential ``dL_t`` has zero mean and no
functional dependence on its own lag (the default GW instrument, ``[1,
dL_{t-h}]``), since the two loss series are built from independent
innovations.

WP-V3 found the h=4 cells here genuinely oversized (Bartlett-tapered HAC +
chi2(q) reference; see ``.dev-notes/anchor_coverage/v3_mc_results.md``
finding 2) -- confirmed NOT a small-n artifact (persisted out to n=100,000).
WP-A1 root-caused it to the linear Bartlett taper discarding a large,
non-vanishing fraction of the *known* (finite-order, exactly h-1-dependent)
autocovariance of an h-step loss differential, and fixed
``giacomini_white_test`` (default ``small_sample=True``) to (a) sum
UNTAPERED autocovariances over the same h-1 lags -- matching ``dm_test``'s
own convention (R's ``forecast::dm.test``) for this exact finite-MA
structure -- with a bandwidth-shrink-on-non-PSD fallback, and (b) reference
``F(q, ESS-q)`` (``ESS = n/(1+2*bandwidth_used)``) instead of chi2(q)
whenever a HAC lag was used. See ``giacomini_white_test``'s docstring for
the full diagnosis and the exact correction. All four cells below (h in
{1,4} x n in {50,200}) are now confirmed in-band with the corrected
default; h=1 is unaffected (bandwidth=0 reduces to the original chi2(q)
reference).
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


@pytest.mark.mc
@pytest.mark.timeout(90)
@pytest.mark.parametrize("n", [50, 200])
@pytest.mark.parametrize("h", [1, 4])
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
            note="default instrument [1, dL_{t-h}]; small_sample=True (chi2(df=2) at h=1, F(q,ESS-q) at h>1)",
            extra={"n": n, "h": h},
        )
        if not in_band:
            violations.append(
                f"alpha={alpha} empirical_rate={n_reject / n_valid:.4f} CI99=[{lo:.4f},{hi:.4f}] ({verdict})"
            )
    assert not violations, f"giacomini_white_test size distortion at n={n}, h={h}: " + "; ".join(violations)
