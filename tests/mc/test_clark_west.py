"""WP-V3 target #2: ``clark_west_test`` size (under the null) and power.

DGP (Clark-West's own motivating nested-model design): a persistent,
irrelevant predictor creates the classic "nested bias" -- the large
(unrestricted) model must estimate an extra parameter that is truly zero,
and that estimation noise inflates its recursive out-of-sample MSPE for
reasons unrelated to true predictive content. A naive test of "which
forecast has the lower MSPE" is therefore biased toward declaring the small
model the winner even when the models are asymptotically equivalent. The
Clark-West (2007) adjustment (``e_r^2 - (e_u^2 - (f_r-f_u)^2)``) exists to
correct exactly this.

  x_t   = rho * x_{t-1} + u_t,           u_t ~ iid N(0,1), rho=0.95
  y_t   = beta1 * x_{t-1} + eps_t,       eps_t ~ iid N(0,1)   (t = 1..T-1)

  small model: recursive/expanding-window mean of y (no predictor)
  large model: recursive/expanding-window OLS of y on [1, x] (nests small)

Under H0 (``beta1=0``), the small model is truly optimal (large model buys
nothing) -- CW's one-sided test (H1: "large model is better") should NOT
over-reject despite the nested bias documented above. Under a moderate
alternative (``beta1=0.05``, tuned to give ~50% power, not a saturated
~100%), CW should have non-trivial power to detect the genuinely better
large model. Both closed-form recursive OLS fits are computed via
cumulative sums (vectorized, no per-origin refit loop) so all the MC cost
is the ``clark_west_test`` call itself, not data generation.
"""
from __future__ import annotations

import numpy as np
import pytest

import macroforecast as mf

from tests.mc.conftest import clopper_pearson, record, spawn_generators

T = 300
R0 = 40
RHO = 0.95
N_REPS_SIZE = 3000
N_REPS_POWER = 1500
ALPHAS = (0.05, 0.10)
POWER_BETA1 = 0.05


def _simulate_nested(
    rng: np.random.Generator, *, beta1: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One replication of the nested AR(1)-predictor DGP; returns (actual, fc_small, fc_large)."""

    x = np.empty(T)
    x[0] = rng.normal(0.0, 1.0 / np.sqrt(max(1e-9, 1.0 - RHO**2)))
    innovations = rng.normal(size=T - 1)
    for t in range(1, T):
        x[t] = RHO * x[t - 1] + innovations[t - 1]
    eps = rng.normal(size=T - 1)
    xk = x[:-1]
    y = beta1 * xk + eps
    m = T - 1
    csum_x = np.cumsum(xk)
    csum_y = np.cumsum(y)
    csum_xx = np.cumsum(xk**2)
    csum_xy = np.cumsum(xk * y)
    n_arr = np.arange(1, m + 1, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta1_hat = (n_arr * csum_xy - csum_x * csum_y) / (n_arr * csum_xx - csum_x**2)
    beta0_hat = (csum_y - beta1_hat * csum_x) / n_arr
    forecast_small_all = csum_y / n_arr
    forecast_large_all = beta0_hat + beta1_hat * x[:m]
    idx = np.arange(R0, m - 1)
    fc_small = forecast_small_all[idx]
    fc_large = forecast_large_all[idx]
    actual = y[idx + 1]
    return actual, fc_small, fc_large


@pytest.mark.mc
@pytest.mark.timeout(120)
def test_clark_west_does_not_overreject_under_null_despite_nested_bias() -> None:
    """H0 true (beta1=0): CW's one-sided ``large model better`` test must not over-reject.

    We also run the *unadjusted* statistic (``cw_adjustment=False``) on the
    exact same replications as a mechanism check: the nested bias should
    make the small model's estimated MSPE look better on average (negative
    mean loss differential), which should make the unadjusted one-sided
    test for "large model wins" reject LESS often than the CW-adjusted one
    (the adjustment corrects the bias, not amplifies it) -- confirming the
    null design actually induces the bias CW is meant to correct, on this
    codebase's own implementation.
    """

    gens = spawn_generators(N_REPS_SIZE, salt=2_000_001)
    pvals_adj = np.empty(N_REPS_SIZE)
    pvals_unadj = np.empty(N_REPS_SIZE)
    mean_loss_diff = np.empty(N_REPS_SIZE)
    for i, rng in enumerate(gens):
        actual, fs, fl = _simulate_nested(rng, beta1=0.0)
        loss_s = (actual - fs) ** 2
        loss_l = (actual - fl) ** 2
        mean_loss_diff[i] = float(np.mean(loss_s - loss_l))
        pvals_adj[i] = mf.tests.clark_west_test(loss_s, loss_l, fs, fl, cw_adjustment=True).p_value
        pvals_unadj[i] = mf.tests.clark_west_test(
            loss_s, loss_l, fs, fl, cw_adjustment=False
        ).p_value

    violations = []
    for alpha in ALPHAS:
        n_reject_adj = int(np.sum(pvals_adj < alpha))
        n_reject_unadj = int(np.sum(pvals_unadj < alpha))
        lo, hi = clopper_pearson(n_reject_adj, N_REPS_SIZE, conf=0.99)
        not_overrejecting = lo <= alpha
        verdict = (
            "PASS (not over-rejecting; conservative under H0 per CW 2007)"
            if lo < alpha
            else ("PASS (in band)" if lo <= alpha <= hi else "OVERSIZED")
        )
        record(
            test="clark_west_test",
            design=(
                f"nested null (beta1=0, rho={RHO}, T={T}, R0={R0}); "
                f"mean(loss_small-loss_large)={float(mean_loss_diff.mean()):.5f} "
                "(negative => nested bias favors small model, as expected)"
            ),
            nominal_alpha=alpha,
            n_reps=N_REPS_SIZE,
            n_rejections=n_reject_adj,
            verdict=verdict,
            note=(
                f"unadjusted (cw_adjustment=False) rate at same alpha="
                f"{n_reject_unadj / N_REPS_SIZE:.4f} for comparison"
            ),
            extra={
                "unadjusted_rate": n_reject_unadj / N_REPS_SIZE,
                "mean_loss_diff": float(mean_loss_diff.mean()),
            },
        )
        if not not_overrejecting:
            violations.append(
                f"alpha={alpha} empirical_rate={n_reject_adj / N_REPS_SIZE:.4f} "
                f"CI99=[{lo:.4f},{hi:.4f}] -- CW OVER-REJECTED under a true null"
            )

    assert not violations, "clark_west_test over-rejected under H0: " + "; ".join(violations)
    # Mechanism check: nested bias should make the small model look better on
    # average (negative mean loss differential), and the CW adjustment should
    # push the "large model wins" rejection rate UP relative to the naive
    # unadjusted statistic (which is biased toward never rejecting in this
    # direction), not down.
    assert mean_loss_diff.mean() < 0.0, (
        "expected the nested-bias DGP to make the small model's estimated "
        f"MSPE look better on average; got mean(loss_small-loss_large)="
        f"{mean_loss_diff.mean():.5f} >= 0"
    )
    assert np.mean(pvals_adj < 0.10) >= np.mean(pvals_unadj < 0.10), (
        "CW adjustment should raise (or leave unchanged), not lower, the "
        "'large model wins' rejection rate relative to the unadjusted "
        "statistic under this nested-bias null"
    )


@pytest.mark.mc
@pytest.mark.timeout(90)
def test_clark_west_has_power_against_a_truly_better_nesting_model() -> None:
    """H1 true (beta1=0.05, tuned for ~50% power): CW should reject at a non-trivial rate.

    Not held to a formal size band (this is a power check, not a size
    check) -- we assert a loose sanity floor well above the nominal alpha to
    catch a functional regression (e.g. a sign flip or an always-fail
    decision rule), and report the point estimate with its CI for
    qualitative comparison to Clark-West (2007)'s own reported power curves.
    """

    gens = spawn_generators(N_REPS_POWER, salt=2_000_002)
    n_reject = 0
    for rng in gens:
        actual, fs, fl = _simulate_nested(rng, beta1=POWER_BETA1)
        loss_s = (actual - fs) ** 2
        loss_l = (actual - fl) ** 2
        result = mf.tests.clark_west_test(loss_s, loss_l, fs, fl)
        if result.decision:
            n_reject += 1

    lo, hi = clopper_pearson(n_reject, N_REPS_POWER, conf=0.99)
    record(
        test="clark_west_test",
        design=f"nested alternative (beta1={POWER_BETA1}, rho={RHO}, T={T}, R0={R0}); large model truly better",
        nominal_alpha=0.05,
        n_reps=N_REPS_POWER,
        n_rejections=n_reject,
        verdict="power reported (not a size check)",
        note="loose floor power > 0.30 asserted as a functional-regression sanity check",
    )
    power = n_reject / N_REPS_POWER
    assert power > 0.30, (
        f"clark_west_test power against a genuinely better nesting model "
        f"(beta1={POWER_BETA1}) is implausibly low: {power:.4f} (CI99="
        f"[{lo:.4f},{hi:.4f}]) -- possible functional regression"
    )
