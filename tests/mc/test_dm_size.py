"""WP-V3 target #1: ``dm_test`` size under genuinely equal predictive accuracy.

DGP: two independent forecast-error series, each an MA(h-1) moving sum of
``h`` iid unit-variance innovations -- ``e_t = eps_t + eps_{t-1} + ... +
eps_{t-h+1}``. This is the textbook representation of an h-step-ahead
recursive forecast error when the one-step innovation is iid: adjacent
origins' errors overlap by ``h-1`` lags, giving the triangular
autocovariance a real h-step DM test must handle (``gamma_k = h-|k|`` for
``|k|<h``, matching a bartlett/acf-kernel HAC's implicit assumption). Both
series use INDEPENDENT innovation draws with identical variance, so
``E[loss_a] == E[loss_b]`` exactly -- H0 (equal predictive accuracy) is true
by construction, at every n/h/correction cell.

We call ``dm_test`` once per replication (kernel="acf", the R
``forecast::dm.test`` default, already parity-checked in
``tests/parity/test_...dm_test...`` -- WP-V1 item 3) and reuse the same
p-value sample to check rejection rate at both alpha=0.05 and alpha=0.10,
since ``TestResult.p_value`` does not depend on the ``alpha`` kwarg.
"""
from __future__ import annotations

import numpy as np
import pytest

import macroforecast as mf

from tests.mc.conftest import clopper_pearson, record, spawn_generators

N_REPS = 3000
ALPHAS = (0.05, 0.10)


def _simulate_ma_errors(rng: np.random.Generator, n: int, h: int) -> np.ndarray:
    """``n`` draws of an MA(h-1) forecast-error series with unit-variance innovations."""

    eps = rng.normal(size=n + h - 1)
    return np.convolve(eps, np.ones(h), mode="valid")


def _dm_pvalues(n: int, h: int, correction: str, *, salt: int) -> np.ndarray:
    gens = spawn_generators(N_REPS, salt=salt)
    pvals = np.empty(N_REPS, dtype=float)
    for i, rng in enumerate(gens):
        loss_a = _simulate_ma_errors(rng, n, h) ** 2
        loss_b = _simulate_ma_errors(rng, n, h) ** 2
        result = mf.tests.dm_test(
            loss_a, loss_b, horizon=h, correction=correction, kernel="acf"
        )
        pvals[i] = result.p_value if result.p_value is not None else np.nan
    return pvals


_KNOWN_DISTORTION_REASON = (
    "CONFIRMED distortion, not a code bug: at n=50,h=4 the acf-kernel long-run "
    "variance estimate uses only 3 autocovariance lags over ~12-13 effective "
    "non-overlapping blocks (n/h), too few for the asymptotic t(n-1) reference "
    "to be accurate. Re-checked at R=10000 (see v3_mc_results.md): "
    "correction='hln' empirical rate 0.074 @alpha=.05 (CI99 [0.067,0.081]) and "
    "0.132 @alpha=.10 (CI99 [0.124,0.141]); correction='none' 0.089 @alpha=.05 "
    "(CI99 [0.082,0.097]) and 0.154 @alpha=.10 (CI99 [0.145,0.164]) -- both "
    "oversized, HLN reduces but does not eliminate the excess (consistent with "
    "Harvey-Leybourne-Newbold 1997's own finding that their correction improves "
    "but does not fully fix size at very small n; see the companion "
    "test_dm_test_hln_correction_closer_to_nominal_at_small_n_large_h, which "
    "asserts and confirms that direction). Diagnosis: variance-estimator small-"
    "sample bias (too few effective blocks for 3 HAC lags at n=50), not a df or "
    "correction-formula defect -- n=200,h=4 (same kernel/correction code path, "
    "more effective blocks) is correctly sized in this same suite."
)

_CASES = [
    pytest.param(50, 1, "hln", id="n50-h1-hln"),
    pytest.param(50, 1, "none", id="n50-h1-none"),
    pytest.param(200, 1, "hln", id="n200-h1-hln"),
    pytest.param(200, 1, "none", id="n200-h1-none"),
    pytest.param(
        50, 4, "hln", id="n50-h4-hln",
        marks=pytest.mark.xfail(reason=_KNOWN_DISTORTION_REASON, strict=True),
    ),
    pytest.param(
        50, 4, "none", id="n50-h4-none",
        marks=pytest.mark.xfail(reason=_KNOWN_DISTORTION_REASON, strict=True),
    ),
    pytest.param(200, 4, "hln", id="n200-h4-hln"),
    pytest.param(200, 4, "none", id="n200-h4-none"),
]


@pytest.mark.mc
@pytest.mark.timeout(120)
@pytest.mark.parametrize("n,h,correction", _CASES)
def test_dm_test_size_equal_accuracy(n: int, h: int, correction: str) -> None:
    salt = 3_000_000 + n * 1000 + h * 10 + (0 if correction == "hln" else 1)
    pvals = _dm_pvalues(n, h, correction, salt=salt)
    valid = pvals[~np.isnan(pvals)]
    n_valid = int(valid.size)
    assert n_valid >= 0.95 * N_REPS, (
        f"too many undefined DM statistics: {N_REPS - n_valid}/{N_REPS} "
        f"(n={n}, h={h}, correction={correction})"
    )
    violations = []
    for alpha in ALPHAS:
        n_reject = int(np.sum(valid < alpha))
        lo, hi = clopper_pearson(n_reject, n_valid, conf=0.99)
        in_band = lo <= alpha <= hi
        verdict = "PASS (in band)" if in_band else ("OVERSIZED" if alpha < lo else "UNDERSIZED")
        record(
            test="dm_test",
            design=f"equal-accuracy MA({h - 1}) errors, n={n}, h={h}, correction={correction}",
            nominal_alpha=alpha,
            n_reps=n_valid,
            n_rejections=n_reject,
            verdict=verdict,
            note="kernel=acf (forecast::dm.test default; parity-checked in WP-V1)",
            extra={"n": n, "h": h, "correction": correction},
        )
        if not in_band:
            violations.append(
                f"alpha={alpha} empirical_rate={n_reject / n_valid:.4f} "
                f"CI99=[{lo:.4f},{hi:.4f}] ({verdict})"
            )
    assert not violations, (
        f"dm_test size distortion: n={n} h={h} correction={correction}: "
        + "; ".join(violations)
    )


@pytest.mark.mc
@pytest.mark.timeout(60)
def test_dm_test_hln_correction_closer_to_nominal_at_small_n_large_h() -> None:
    """HLN (1997): the small-sample correction should reduce over-rejection.

    At small n and h>1 the asymptotic (uncorrected) DM test is known to
    over-reject because the HAC long-run variance estimate is noisy relative
    to the mean; the HLN correction shrinks the statistic to compensate. We
    assert the DIRECTION documented in Harvey-Leybourne-Newbold (1997): at
    n=50, h=4, the HLN-corrected empirical rejection rate must be at least as
    close to the nominal alpha as the uncorrected rate, for both alpha=0.05
    and alpha=0.10.
    """

    n, h = 50, 4
    pvals_hln = _dm_pvalues(n, h, "hln", salt=3_000_000 + n * 1000 + h * 10 + 0)
    pvals_none = _dm_pvalues(n, h, "none", salt=3_000_000 + n * 1000 + h * 10 + 1)
    valid_hln = pvals_hln[~np.isnan(pvals_hln)]
    valid_none = pvals_none[~np.isnan(pvals_none)]

    for alpha in ALPHAS:
        rate_hln = float(np.mean(valid_hln < alpha))
        rate_none = float(np.mean(valid_none < alpha))
        dist_hln = abs(rate_hln - alpha)
        dist_none = abs(rate_none - alpha)
        record(
            test="dm_test_hln_direction",
            design=f"n={n}, h={h}, hln vs none",
            nominal_alpha=alpha,
            n_reps=int(valid_hln.size),
            n_rejections=int(np.sum(valid_hln < alpha)),
            verdict="PASS (HLN closer or equal)" if dist_hln <= dist_none else "DIRECTION VIOLATED",
            note=f"rate_hln={rate_hln:.4f} rate_none={rate_none:.4f}",
            extra={"rate_hln": rate_hln, "rate_none": rate_none},
        )
        assert dist_hln <= dist_none, (
            f"HLN correction did not reduce size distortion at n={n}, h={h}, "
            f"alpha={alpha}: |rate_hln-alpha|={dist_hln:.4f} > "
            f"|rate_none-alpha|={dist_none:.4f} (rate_hln={rate_hln:.4f}, "
            f"rate_none={rate_none:.4f})"
        )
