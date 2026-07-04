"""``macroforecast.tests._berkowitz_density_test`` (reached via the public
``density_interval_tests`` wrapper) vs a clean-room R implementation of
Berkowitz (2001) -- work item 5 of the WP-V1 brief.

No canonical CRAN package implements the Berkowitz (2001) density-forecast
test directly (the docstring's own claimed alignment,
``tstests/R/berkowitz.R::berkowitz_test``, is a GitHub gist/course-note
script, not a CRAN package we can `install.packages()`), so per the WP-V1
brief this is a clean-room R port of the same 3-step closed-form procedure
macroforecast documents for itself:

  1. PIT values -> qnorm (inverse normal CDF) transform to get ``z``.
  2. Fit AR(``lags``,0,0) with intercept to ``z`` (macroforecast:
     ``statsmodels.tsa.arima.model.ARIMA(z, order=(lags,0,0), trend="c")``;
     R: base ``stats::arima(z, order=c(lags,0,0), include.mean=TRUE)`` --
     both are Kalman-filter state-space Gaussian MLE fits of the same
     model, from two different software implementations).
  3. LR statistic = -2 * (restricted N(0,1) log-likelihood - unrestricted
     AR(lags) log-likelihood), chi-squared with df = 2 + lags.

Tolerance: this is an MLE-through-an-optimizer comparison across two
different software stacks (statsmodels' state-space Kalman filter vs R's
``arima()`` state-space Kalman filter) rather than a shared closed form, so
exact bit-parity is not expected. We use rel=1e-3 on the LR statistic
(loose enough to absorb optimizer/starting-value differences between the
two packages, tight enough that a real formula error -- e.g. wrong df, or
a sign error in the LR expression -- would still fail it) and check the
resulting p-value and reject-decision at alpha=0.05 for full agreement
(the substantive claim of the test), which is much less sensitive to small
LR differences than the raw LR value itself.
"""
from __future__ import annotations

import numpy as np
import pytest

import macroforecast as mf

from tests.parity.conftest import parse_float, require_r, run_rscript, write_csv

pytestmark = [pytest.mark.rparity]


def _r_berkowitz(pit: np.ndarray, *, lags: int, tmp_path) -> dict[str, float]:
    csv_path = tmp_path / "pit.csv"
    write_csv(csv_path, {"pit": pit})
    script = f'''
df <- read.csv("{csv_path}")
pit <- pmin(pmax(df$pit, 1e-9), 1 - 1e-9)
z <- qnorm(pit)
n <- length(z)
restricted_ll <- sum(dnorm(z, mean = 0, sd = 1, log = TRUE))
fit <- arima(z, order = c({lags}, 0, 0), include.mean = TRUE)
unrestricted_ll <- as.numeric(fit$loglik)
lr <- max(-2 * (restricted_ll - unrestricted_ll), 0)
df_chisq <- 2 + {lags}
p_value <- pchisq(lr, df = df_chisq, lower.tail = FALSE)
emit("lr", lr)
emit("p_value", p_value)
emit("df", df_chisq)
'''
    result = run_rscript(script)
    return {
        "lr": parse_float(result["lr"]),
        "p_value": parse_float(result["p_value"]),
        "df": parse_float(result["df"]),
    }


def _pit_fixture(n: int = 250, seed: int = 5, ar_coef: float = 0.3) -> np.ndarray:
    """PIT values with a mild AR(1) dependence in normal-quantile space, so
    the unrestricted AR(1) fit genuinely differs from the restricted N(0,1)
    (a fixture of pure iid N(0,1) PITs would make both models coincide and
    the LR statistic trivially ~0, which would not exercise the formula).
    """
    from scipy import stats as scipy_stats

    rng = np.random.default_rng(seed)
    z = np.empty(n)
    z[0] = rng.standard_normal()
    for t in range(1, n):
        z[t] = ar_coef * z[t - 1] + rng.standard_normal()
    return scipy_stats.norm.cdf(z)


def test_berkowitz_lr_statistic_and_pvalue_match_r_clean_room(tmp_path) -> None:
    require_r()
    pit = _pit_fixture()

    r = _r_berkowitz(pit, lags=1, tmp_path=tmp_path)
    py_result = mf.tests.density_interval_tests(pit, pit_lag=1, alpha=0.05)
    py_berkowitz = py_result["berkowitz"]

    assert py_berkowitz["df"] == int(r["df"])
    assert py_berkowitz["lr_statistic"] == pytest.approx(r["lr"], rel=1e-3), (
        f"LR statistic mismatch: py={py_berkowitz['lr_statistic']!r} vs R={r['lr']!r}"
    )
    assert py_berkowitz["p_value"] == pytest.approx(r["p_value"], rel=1e-3, abs=1e-6), (
        f"p-value mismatch: py={py_berkowitz['p_value']!r} vs R={r['p_value']!r}"
    )
    assert py_berkowitz["reject"] == bool(r["p_value"] < 0.05), (
        "reject decision at alpha=0.05 disagrees between macroforecast and R"
    )


def test_berkowitz_null_fixture_fails_to_reject_in_both(tmp_path) -> None:
    """Sanity/negative control: PIT values genuinely drawn iid N(0,1)
    (ar_coef=0) should NOT reject the null in either implementation --
    guards against a test that "passes" only because both sides always
    reject regardless of the data.
    """
    require_r()
    pit = _pit_fixture(seed=11, ar_coef=0.0)

    r = _r_berkowitz(pit, lags=1, tmp_path=tmp_path)
    py_result = mf.tests.density_interval_tests(pit, pit_lag=1, alpha=0.05)
    py_berkowitz = py_result["berkowitz"]

    assert r["p_value"] > 0.05, "negative control: R itself rejected under the null fixture"
    assert py_berkowitz["p_value"] > 0.05, "negative control: macroforecast rejected under the null fixture"
    assert py_berkowitz["reject"] is False
