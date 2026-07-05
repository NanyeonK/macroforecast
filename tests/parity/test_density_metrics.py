"""Exact-value oracles for the Tier-1 density/proper-scoring metrics flagged
in the WP-V0 inventory as ``np.isfinite``-only coverage: ``gaussian_nll``,
``crps``, ``log_score``, ``negative_log_score``, ``qlike``, ``smape`` --
work item 6 of the WP-V1 brief.

Two kinds of oracle, matching the brief's ask ("hand-computed toy fixtures
in the same style as the existing pinball_loss tests, plus scoringRules
parity for crps"):

- ``gaussian_nll``/``log_score``/``negative_log_score`` (all the same
  Gaussian NLL under different names -- confirmed by reading
  macroforecast/metrics.py: both aliases literally ``return
  gaussian_nll(...)``) are checked against ``scipy.stats.norm.logpdf``, an
  independent, already-tested Gaussian log-density implementation neither
  written nor maintained by this package -- a genuine external oracle, not
  a restatement of macroforecast's own formula.
- ``qlike`` and ``smape`` have simple enough closed forms that hand-typed
  literal expected values (no formula re-typed in the test, just the
  arithmetic result) are a real independent check, in the same spirit as
  the existing ``test_distribution_metric_helpers_validate_inputs``
  (``pinball_loss([1, 2], [0, 3], quantile=0.5) == 0.5``).
- ``crps`` is checked against R ``scoringRules::crps_norm`` (closed-form
  Gaussian CRPS, tight tolerance) AND ``scoringRules::crps_sample`` (a
  Monte-Carlo/edf estimator from a large fixed-seed sample drawn from the
  same Gaussian predictive distribution -- looser tolerance, justified by
  its O(1/sqrt(N)) Monte Carlo error).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as scipy_stats

import macroforecast as mf

from tests.parity.conftest import parse_float, require_r, run_rscript, write_csv

pytestmark = [pytest.mark.rparity]


def test_qlike_hand_computed_toy_values() -> None:
    # qlike = mean(log(forecast) + realized / forecast); realized=2, forecast=1
    # -> log(1) + 2/1 = 0 + 2 = 2.0
    assert mf.metrics.qlike([2.0], [1.0]) == pytest.approx(2.0)
    # realized=1, forecast=1 -> log(1) + 1/1 = 0 + 1 = 1.0 (the metric's
    # minimum-at-perfect-forecast-of-1 case)
    assert mf.metrics.qlike([1.0], [1.0]) == pytest.approx(1.0)
    # two obs: [2,8] realized vs forecast [1,4] -> (0+2) and (log(4)+2) averaged
    expected = np.mean([np.log(1.0) + 2.0 / 1.0, np.log(4.0) + 8.0 / 4.0])
    assert mf.metrics.qlike([2.0, 8.0], [1.0, 4.0]) == pytest.approx(expected)


def test_smape_hand_computed_toy_values_including_200_ceiling() -> None:
    # |1-3|=2, denom=max((1+3)/2, eps)=2, ratio=1 -> 100.0
    assert mf.metrics.smape([1.0], [3.0]) == pytest.approx(100.0)
    # perfect forecast -> 0
    assert mf.metrics.smape([4.0], [4.0]) == pytest.approx(0.0)
    # opposite signs of equal magnitude: |(-2)-2|=4, denom=max((2+2)/2,eps)=2,
    # ratio=2 -> 200.0, the documented ceiling of the 0-200 M4 sMAPE scale.
    assert mf.metrics.smape([-2.0], [2.0]) == pytest.approx(200.0)


@pytest.mark.parametrize("fn", ["gaussian_nll", "log_score", "negative_log_score"])
def test_gaussian_nll_family_matches_scipy_normal_logpdf(fn: str) -> None:
    rng = np.random.default_rng(3)
    y_true = rng.normal(size=12)
    y_pred = rng.normal(size=12)
    variance = rng.uniform(0.2, 3.0, size=12)

    expected = float(
        np.mean(-scipy_stats.norm.logpdf(y_true, loc=y_pred, scale=np.sqrt(variance)))
    )
    actual = getattr(mf.metrics, fn)(y_true, y_pred, variance)
    assert actual == pytest.approx(expected, rel=1e-12)


def _r_crps(y_true: np.ndarray, y_pred: np.ndarray, variance: np.ndarray, tmp_path) -> dict[str, list[float]]:
    csv_path = tmp_path / "gauss.csv"
    write_csv(csv_path, {"y": y_true, "mean": y_pred, "sd": np.sqrt(variance)})
    script = f'''
library(scoringRules)
df <- read.csv("{csv_path}")
crps_closed <- crps_norm(df$y, mean = df$mean, sd = df$sd)
emit("crps_norm", crps_closed)

set.seed(2024)
n_sample <- 20000
crps_mc <- numeric(nrow(df))
for (i in seq_len(nrow(df))) {{
  draws <- rnorm(n_sample, mean = df$mean[i], sd = df$sd[i])
  crps_mc[i] <- crps_sample(df$y[i], draws)
}}
emit("crps_sample", crps_mc)
'''
    result = run_rscript(script, timeout=120)
    return {
        "crps_norm": [parse_float(v) for v in result["crps_norm"].split(",")],
        "crps_sample": [parse_float(v) for v in result["crps_sample"].split(",")],
    }


def test_crps_matches_scoringrules_crps_norm_closed_form(tmp_path) -> None:
    require_r("scoringRules")
    rng = np.random.default_rng(9)
    y_true = rng.normal(size=8)
    y_pred = rng.normal(size=8)
    variance = rng.uniform(0.2, 3.0, size=8)

    r = _r_crps(y_true, y_pred, variance, tmp_path)

    for i in range(len(y_true)):
        py_val = mf.metrics.crps([y_true[i]], [y_pred[i]], [variance[i]])
        assert py_val == pytest.approx(r["crps_norm"][i], rel=1e-6), (
            f"crps[{i}] closed-form mismatch: py={py_val!r} vs R crps_norm={r['crps_norm'][i]!r}"
        )


def test_crps_matches_scoringrules_crps_sample_monte_carlo(tmp_path) -> None:
    """Monte-Carlo cross-check: draw 20,000 samples from the SAME Gaussian
    predictive distribution in R and score with scoringRules::crps_sample
    (an edf/energy-distance estimator, not the closed form). Tolerance is
    looser (abs=0.02) to absorb O(1/sqrt(20000)) ~ 0.007-scale Monte Carlo
    error on top of the R/Python RNG streams being independent draws from
    the same distribution, not shared numbers.
    """
    require_r("scoringRules")
    rng = np.random.default_rng(9)
    y_true = rng.normal(size=8)
    y_pred = rng.normal(size=8)
    variance = rng.uniform(0.2, 3.0, size=8)

    r = _r_crps(y_true, y_pred, variance, tmp_path)

    for i in range(len(y_true)):
        py_val = mf.metrics.crps([y_true[i]], [y_pred[i]], [variance[i]])
        assert py_val == pytest.approx(r["crps_sample"][i], abs=0.02), (
            f"crps[{i}] vs Monte Carlo crps_sample: py={py_val!r} vs R crps_sample={r['crps_sample'][i]!r}"
        )
