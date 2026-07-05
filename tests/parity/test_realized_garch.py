"""``macroforecast.models.realized_garch`` vs R ``rugarch::ugarchfit(model=
"realGARCH")`` -- restores/adapts XR-3 of the deleted C59 suite
(``git show 7050f5d2:tests/core/test_r_crossref_c59.py``), work item 1 of the
WP-V1 brief.

Two things changed enough since the deleted test that it needed a rewrite
rather than a straight port, both discovered while adapting it (not assumed):

1. API: the deleted test called a ``_build_l4_model("realized_garch",
   params=...)``/``.fit(X, y)`` interface that no longer exists.
   Current API: ``macroforecast.models.realized_garch(y, rv=..., ...)``
   returning a ``VolatilityFit`` whose ``.diagnostics["params"]`` dict has
   the fitted parameters (see ``macroforecast/models/volatility.py``,
   ``RealizedGARCHEstimator``).

2. Parametrization: the deleted test's DGP (11 true params incl. ``tau_1``,
   ``tau_2``, ``gamma``, ``phi``, separate ``delta_1``/``delta_2``) does not
   match the CURRENT ``RealizedGARCHEstimator``, which implements a
   simpler, compact 9-parameter recursion (see the class docstring/comments
   in ``macroforecast/models/volatility.py``):
     log h_t = omega + alpha * log(x_{t-1}) + beta * log(h_{t-1})
     z_t = (r_t - mu) / sqrt(h_t)
     tau(z_t) = eta_1 * z_t + eta_2 * (z_t**2 - 1)
     log x_t = xi + delta * log(h_t) + tau(z_t) + u_t,  u_t ~ N(0, sigma_u**2)
   Feeding the deleted test's old DGP into the current estimator would not
   be testing the model that exists today, so this file simulates from
   the CURRENT model's own documented recursion instead.

Cross-checked against rugarch's own vignette (``Introduction_to_the_
rugarch_package.pdf``, section 2.2.9, equations 46-47 -- extracted via
``pdftotext`` and read directly, not assumed), the current
``RealizedGARCHEstimator`` recursion is essentially IDENTICAL, term for
term, to rugarch's realGARCH(1,1) with a documented 1:1 name
correspondence (rugarch name -> macroforecast name): mu->mu, omega->omega,
alpha1->alpha, beta1->beta, xi->xi, delta->delta, eta11->eta_1,
eta21->eta_2, lambda->sigma_u. Because of this exact correspondence, this
test compares ALL NINE overlapping parameters (not just the 3 the deleted
test compared), not only the weaker ``atol``-per-parameter subset.

One nontrivial, empirically-verified (not assumed) calling-convention gotcha
surfaced while building this: rugarch's ``realizedVol`` argument to
``ugarchfit``/``ugarchfilter`` -- despite its name -- is used DIRECTLY as
the log-linear measurement-equation series (rugarch's "rt" in the vignette
notation), NOT squared internally. Confirmed via a controlled experiment:
``ugarchfilter`` with ``fixed.pars`` (no optimization at all) reproduces a
hand-rolled Python recursion using ``realizedVol`` as-is to within 5e-8,
while a "square it first" hypothesis is off by up to 0.38 in absolute
terms on the same data. So this file passes macroforecast's ``x`` (the
realized measure fed via ``rv=``) directly as rugarch's ``realizedVol=``,
with NO sqrt/square transform in between -- getting this wrong (as an
initial exploratory pass here did, passing ``sqrt(x)``) silently produces
a self-consistent-looking fit with every measurement-equation coefficient
off by a clean factor of ~2 (or ~0.5), which would misdiagnose a pure
calling-convention error as a model bug.

Also confirmed empirically (not assumed): rugarch's ``lambda`` coefficient,
despite the vignette writing "u_t ~ N(0, lambda)" (suggesting lambda is a
variance), numerically equals macroforecast's ``sigma_u`` (the STANDARD
DEVIATION) almost exactly in both fixtures below -- not ``sigma_u**2``. This
is reported as observed rather than re-derived from the variance notation.

Tolerance: atol=0.01 per parameter, chosen from two independent
T=1500, seed={42, 99} simulations of this exact recursion, both of which
matched to within ~0.002 on every one of the 9 parameters (an MLE-through-
two-different-optimizers comparison) -- atol=0.01 keeps roughly 5x margin
over the observed agreement while still being tight enough that a genuine
formula divergence (which this harness would have caught, per the exact
correspondence derived from rugarch's own vignette) would fail loudly.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf

from tests.parity.conftest import parse_float, require_r, run_rscript, write_csv

pytestmark = [pytest.mark.rparity]

_TRUE_PARAMS = {
    "mu": 0.03,
    "omega": 0.05,
    "alpha": 0.40,
    "beta": 0.55,
    "xi": -0.10,
    "delta": 0.90,
    "eta_1": -0.05,
    "eta_2": 0.03,
    "sigma_u": 0.30,
}

_ATOL = 0.01


def _simulate(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Simulate directly from RealizedGARCHEstimator's own documented
    recursion (macroforecast/models/volatility.py, RealizedGARCHEstimator
    docstring/comments) -- confirmed term-for-term identical to rugarch
    realGARCH(1,1) via the vignette equations (46)-(47).
    """
    rng = np.random.default_rng(seed)
    p = _TRUE_PARAMS
    h = np.zeros(n)
    x = np.zeros(n)
    r = np.zeros(n)
    log_h_prev = math.log(1.0)
    log_x_prev = math.log(1.0)
    for t in range(n):
        log_h = p["omega"] + p["alpha"] * log_x_prev + p["beta"] * log_h_prev
        h[t] = math.exp(np.clip(log_h, -30.0, 30.0))
        z = rng.standard_normal()
        r[t] = p["mu"] + math.sqrt(h[t]) * z
        tau = p["eta_1"] * z + p["eta_2"] * (z**2 - 1.0)
        u = p["sigma_u"] * rng.standard_normal()
        log_x = p["xi"] + p["delta"] * math.log(h[t]) + tau + u
        x[t] = math.exp(log_x)
        log_h_prev = math.log(h[t])
        log_x_prev = log_x
    return pd.DataFrame({"r": r, "x": x})


def _r_realized_garch(df: pd.DataFrame, tmp_path) -> dict[str, float]:
    csv_path = tmp_path / "realgarch.csv"
    df.to_csv(csv_path, index=False)
    script = f'''
suppressMessages(library(rugarch))
suppressMessages(library(xts))
df <- read.csv("{csv_path}")
n <- nrow(df)
dates <- seq(as.Date("2000-01-01"), by = "day", length.out = n)
r_xts <- xts(df$r, order.by = dates)
# IMPORTANT (see module docstring): realizedVol is used AS-IS, not squared.
rv_xts <- xts(df$x, order.by = dates)
spec <- ugarchspec(
  variance.model = list(model = "realGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
  distribution.model = "norm"
)
fit <- ugarchfit(spec = spec, data = r_xts, realizedVol = rv_xts, solver = "hybrid")
cf <- coef(fit)
emit("mu", cf[["mu"]])
emit("omega", cf[["omega"]])
emit("alpha", cf[["alpha1"]])
emit("beta", cf[["beta1"]])
emit("xi", cf[["xi"]])
emit("delta", cf[["delta"]])
emit("eta_1", cf[["eta11"]])
emit("eta_2", cf[["eta21"]])
emit("sigma_u", cf[["lambda"]])
'''
    result = run_rscript(script, timeout=180)
    return {key: parse_float(result[key]) for key in _TRUE_PARAMS}


@pytest.mark.parametrize("seed", [42, 99])
def test_realized_garch_params_match_rugarch_realgarch(seed: int, tmp_path) -> None:
    require_r("rugarch")
    df = _simulate(n=1500, seed=seed)

    fit = mf.models.realized_garch(df["r"], rv=df["x"], max_iter=3000, n_starts=8, random_state=0)
    py_params = dict(fit.diagnostics["params"])
    py_params["sigma_u"] = math.exp(py_params["log_sigma_u"])

    r_params = _r_realized_garch(df, tmp_path)

    failures = []
    for name in _TRUE_PARAMS:
        py_val = py_params[name]
        r_val = r_params[name]
        if abs(py_val - r_val) > _ATOL:
            failures.append(f"{name}: py={py_val!r} vs R={r_val!r} (atol={_ATOL})")
    assert not failures, "realized_garch vs rugarch::realGARCH mismatch:\n" + "\n".join(failures)


def test_realized_garch_recovers_true_dgp_params(tmp_path) -> None:
    """Sanity/negative control: macroforecast's own MLE should recover the
    TRUE simulated parameters reasonably well -- otherwise a match with R
    could mean "both wrong in the same way" rather than "both correct."
    """
    df = _simulate(n=1500, seed=42)
    fit = mf.models.realized_garch(df["r"], rv=df["x"], max_iter=3000, n_starts=8, random_state=0)
    py_params = dict(fit.diagnostics["params"])
    py_params["sigma_u"] = math.exp(py_params["log_sigma_u"])

    failures = []
    for name, true_val in _TRUE_PARAMS.items():
        if abs(py_params[name] - true_val) > 0.06:
            failures.append(f"{name}: fitted={py_params[name]!r} vs true={true_val!r}")
    assert not failures, "realized_garch failed to recover its own DGP's true parameters:\n" + "\n".join(failures)
