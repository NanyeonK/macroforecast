"""``macroforecast.models.restricted_midas``/``midas_almon`` vs R
``midasr::midas_r(weight_method="nealmon")`` -- restores/adapts XR-2 of the
deleted C59 suite (``git show 7050f5d2:tests/core/test_r_crossref_c59.py``),
work item 1 of the WP-V1 brief.

The repo restructure since C59 split what used to be one MIDAS callable
into two genuinely different estimators, discovered while adapting this
test (not assumed) by reading ``macroforecast/models/timeseries.py``:

- ``restricted_midas(weighting="almon")`` (backed by
  ``_RestrictedMIDASRegressor``) jointly estimates the nealmon shape
  parameters AND the aggregate scale/intercept by nonlinear least squares
  (``scipy.optimize.least_squares``) -- its own source comment says so
  explicitly: "R source alignment, midasr/R/midasreg.R ... midasr::midas_r
  jointly estimates restricted lag-weight parameters and regression
  coefficients by nonlinear least squares. This estimator follows the
  same objective." Its parametrization is a LINE-FOR-LINE match to
  ``midasr:::nealmon``'s own source (``print(midasr::nealmon)``, read
  directly): ``p[1]`` is a free scale multiplying the normalized
  exp-Almon weight shape, ``p[-1]`` (length = polynomial_order) are the
  polynomial coefficients inside the exponential -- and
  ``_restricted_midas_weights``'s "almon" branch requires exactly
  ``1 + polynomial_order`` params with an identical scale-times-normalized-
  exponential formula. THIS is the genuinely comparable-to-midasr callable.
- ``midas_almon`` (backed by ``_MIDASRegressor``) instead takes theta as a
  FIXED hyperparameter (default all-zero, i.e. flat/equal weights unless
  the caller supplies one) and estimates only a linear/ridge aggregate
  coefficient on top -- its own docstring says "macroforecast estimates
  the aggregate coefficient with a linear/ridge head rather than midasr's
  joint NLS optimizer." This is a deliberate, documented architecture
  difference, not a bug -- see
  ``test_midas_almon_fixed_theta_diverges_from_midasr_joint_nls`` below for
  the executable confirmation that its predictions genuinely differ from
  midasr's jointly-optimized fit (xfail, not silently dropped).

Frequency handling: rather than reverse-engineer a full monthly/quarterly
``fmls(x, k, m)`` alignment (``m`` = frequency ratio) for this restructured
API, this file uses ``m=1`` (same-frequency, ``k`` ordinary lags) --
confirmed empirically that ``midasr::fmls(x, k=3, m=1)`` produces columns
``X.0/m, X.1/m, X.2/m, X.3/m`` that are exactly ``x[t], x[t-1], x[t-2],
x[t-3]`` (contemporaneous plus 3 lags), which is precisely macroforecast's
own ``x_lag0, x_lag1, x_lag2, x_lag3`` wide-lag-column convention. This
sidesteps the frequency-mismatch machinery (out of scope for this WP) while
still exercising the identical nealmon weight-restriction/NLS-estimation
core that XR-2 was written to check.

Tolerance: atol=0.01 per parameter and atol=0.01 on fitted values (absolute,
not relative -- several fitted values in this fixture cross zero, where a
relative tolerance is not meaningful), from two independent T=200,
seed={7, 123} simulations both matching to within ~0.002 on every one of
the 4 parameters (intercept, scale, theta1, theta2) and ~0.004 on fitted
values -- an NLS-through-two-different-optimizers comparison (SciPy
``least_squares`` vs R's ``optim`` default for ``midas_r``), so not
exact-formula tight, but with ~2.5x margin over observed fitted-value
agreement (consistent with propagating the ~0.01-scale parameter
differences through the same weighted-sum design).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf

from tests.parity.conftest import parse_float, parse_float_list, require_r, run_rscript, write_csv

pytestmark = [pytest.mark.rparity]

_K = 3  # lags 0..3 (4 lag columns)


def _build_fixture(
    n: int = 200,
    seed: int = 7,
    true_theta: tuple[float, float] = (0.3, -0.05),
    true_scale: float = 2.0,
    true_intercept: float = 1.0,
    noise_sd: float = 0.05,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    rng = np.random.default_rng(seed)
    lead = _K
    x_raw = rng.standard_normal(n + lead)
    positions = np.arange(1, _K + 2, dtype=float)
    raw = true_theta[0] * positions + true_theta[1] * positions**2
    exp_raw = np.exp(raw - raw.max())
    weight_shape = exp_raw / exp_raw.sum()

    lag_cols = {f"x_lag{i}": x_raw[lead - i : lead - i + n] for i in range(_K + 1)}
    lagged = pd.DataFrame(lag_cols)
    weighted_sum = sum(weight_shape[i] * lagged[f"x_lag{i}"] for i in range(_K + 1))
    y = true_intercept + true_scale * weighted_sum + rng.standard_normal(n) * noise_sd
    return lagged, pd.Series(y, name="y"), x_raw


def _r_midas_r(x_raw: np.ndarray, y: pd.Series, tmp_path) -> dict:
    n = len(y)
    lead = _K
    x_path = tmp_path / "x.csv"
    y_path = tmp_path / "y.csv"
    write_csv(x_path, {"x": x_raw})
    write_csv(y_path, {"y": y.to_numpy()})
    script = f'''
suppressMessages(library(midasr))
x <- read.csv("{x_path}")$x
y <- read.csv("{y_path}")$y
n <- {n}
lead <- {lead}
x_aligned <- x[(lead + 1):(lead + n)]
fit <- midas_r(y ~ fmls(x_aligned, {_K}, 1, nealmon), start = list(x_aligned = c(1, 0, 0)))
cf <- coef(fit)
emit("intercept", cf[["(Intercept)"]])
emit("scale", cf[["x_aligned1"]])
emit("theta1", cf[["x_aligned2"]])
emit("theta2", cf[["x_aligned3"]])
emit("fitted", fitted(fit))
'''
    result = run_rscript(script, timeout=120)
    return {
        "intercept": parse_float(result["intercept"]),
        "scale": parse_float(result["scale"]),
        "theta1": parse_float(result["theta1"]),
        "theta2": parse_float(result["theta2"]),
        "fitted": parse_float_list(result["fitted"]),
    }


@pytest.mark.parametrize(
    "seed,true_theta,true_scale,true_intercept",
    [
        (7, (0.3, -0.05), 2.0, 1.0),
        (123, (0.2, -0.08), 1.5, -0.5),
    ],
)
def test_restricted_midas_almon_matches_midasr_nealmon(
    seed: int, true_theta: tuple[float, float], true_scale: float, true_intercept: float, tmp_path
) -> None:
    require_r("midasr")
    lagged, y, x_raw = _build_fixture(
        seed=seed, true_theta=true_theta, true_scale=true_scale, true_intercept=true_intercept
    )

    fit = mf.models.restricted_midas(
        lagged, y, weighting="almon", polynomial_order=2,
        start_params=(1.0, 0.0, 0.0), maxiter=2000, tolerance=1e-10,
    )
    py_params = fit.diagnostics["restricted_parameters"]

    r = _r_midas_r(x_raw, y, tmp_path)

    assert py_params["intercept"] == pytest.approx(r["intercept"], abs=0.01)
    assert py_params["x_theta0"] == pytest.approx(r["scale"], abs=0.01)
    assert py_params["x_theta1"] == pytest.approx(r["theta1"], abs=0.01)
    assert py_params["x_theta2"] == pytest.approx(r["theta2"], abs=0.01)

    # Fitted-value comparison: R's fitted() drops the first `_K` rows (fmls
    # generates NA there), so compare macroforecast's predictions on the
    # same offset slice of `lagged`.
    py_pred = np.asarray(fit.predict(lagged))[_K:]
    r_pred = np.asarray(r["fitted"])
    assert py_pred == pytest.approx(r_pred, abs=0.01)


def test_midas_almon_fixed_theta_diverges_from_midasr_joint_nls(tmp_path) -> None:
    """Documented architecture-deviation finding, not a bug: ``midas_almon``
    fixes theta (default 0, i.e. flat weights) and only fits a linear/ridge
    aggregate coefficient, so its predictions should NOT be expected to
    match ``midasr::midas_r``'s jointly-NLS-optimized fit whenever the true
    theta is non-trivial (as in this fixture). Confirms the divergence
    exists (xfail-style informational check) rather than silently omitting
    the comparison the deleted test's XR-2 originally made against
    ``midas_almon`` under its old, since-removed API.
    """
    require_r("midasr")
    lagged, y, x_raw = _build_fixture(seed=7)

    almon_fit = mf.models.midas_almon(lagged, y, polynomial_order=2)
    py_pred = np.asarray(almon_fit.predict(lagged))[_K:]

    r = _r_midas_r(x_raw, y, tmp_path)
    r_pred = np.asarray(r["fitted"])

    max_rel_dev = float(np.max(np.abs(py_pred - r_pred) / (np.abs(r_pred) + 1e-8)))
    # This is an assertion that the deviation IS large (confirming the
    # architecture-difference diagnosis), not that predictions match.
    assert max_rel_dev > 0.10, (
        "midas_almon (fixed theta=0) unexpectedly matched midasr's jointly-"
        f"optimized fit closely (max_rel_dev={max_rel_dev!r} <= 0.10) -- if "
        "this starts failing, midas_almon's default behavior changed and "
        "this documented architecture-deviation finding needs re-diagnosis."
    )
