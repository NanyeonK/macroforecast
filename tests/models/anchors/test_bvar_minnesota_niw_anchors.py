"""Independent correctness anchors for bvar_minnesota / bvar_normal_inverse_wishart.

WP-V2 (model-anchors-v2). Both models were Tier-1 zero-anchor in
``.dev-notes/anchor_coverage/summary.md``: the FAVAR::BVAR-aligned Gibbs
sampler (``_favar_bvar_draws`` / ``_draw_bvar_coefficients`` in
``macroforecast.models.timeseries``) had never been checked against any
independent reference -- only a known-DGP own-lag-recovery oracle and a
prior-scale regression pin existed.

Two anchors here, both derived from first principles rather than by reading
the Gibbs sampler and copying its formula:

1. ``test_minnesota_prior_variance_matches_documented_formula`` (deterministic,
   1e-12): the own-lag / cross-lag / lag-decay pattern that
   ``bvar_minnesota``'s docstring claims alignment with
   (``bvartools::minnesota_prior``) is, per that R function's own documented
   formula (Chan, Koop, Poirier & Tobias 2020; Litterman 1986):

       var[eq, lag, reg] = kappa0 / lag**2                                  (eq == reg)
       var[eq, lag, reg] = kappa0 * kappa1 / lag**2 * sigma_eq**2/sigma_reg**2  (eq != reg)

   where sigma_i is the residual SD of an unrestricted univariate AR(n_lag)
   OLS fit of series i. This test reimplements that formula independently
   (own AR-OLS fit, own nested loop) and compares to
   ``macroforecast.models.timeseries._favar_minnesota_prior``'s output.

2. ``test_niw_gibbs_posterior_mean_matches_closed_form_ols`` (Monte Carlo,
   fixed seed): with the diffuse ``bvar_normal_inverse_wishart`` defaults
   (b0=0, vb0=0), the prior precision on vec(A) is the zero matrix. Every
   Gibbs sweep's conditional posterior mean for A, given ANY drawn Sigma, is
   then the plain (generalized-least-squares) Bayesian-linear-regression
   update with a flat prior. Because every equation of a VAR shares the
   IDENTICAL design matrix X, the GLS estimator is algebraically identical
   to the equation-by-equation OLS estimator for *any* positive-definite
   error covariance Sigma (the classical SUR-with-common-regressors
   equivalence, e.g. Zellner 1962 sec. 4). So the Gibbs sampler's posterior
   mean must converge (Monte Carlo sense) to plain OLS on the shared VAR
   design -- a closed-form target computed independently of the sampler's
   own code path. The tolerance is set from the sampler's own reported Monte
   Carlo standard error (``coef_mcse`` in diagnostics), not an arbitrarily
   loosened number.

A companion live R cross-check of anchor (1) against ``bvartools::minnesota_prior``
(the exact R function the docstring names) lives in
``tests/parity/test_bvar_minnesota_r_parity.py``.

FIX (WP-V2 follow-up, same branch): the near-singular-residual-covariance
finding below was originally an ``xfail(strict=True)`` regression pin. It is
now fixed -- ``bvar_normal_inverse_wishart``/``bvar_minnesota``'s ``s0``
default changed from the exactly-zero ``0.0`` to ``None``, which resolves to
a data-dependent diagonal scale ``diag(sigma_1**2, ..., sigma_k**2)`` from
each series' own AR(n_lag)-OLS residual variance (see
``_favar_default_niw_scale`` in ``macroforecast.models.timeseries``, and the
CHANGELOG ``[Unreleased]`` entry). A divergence guard
(``_warn_if_bvar_draws_diverged``) also now fires a ``UserWarning`` whenever
post-burn-in draws explode, which fires for the *old* behavior (an
explicitly passed ``s0=0.0``) so that path stays non-silent too. The finding
below is kept for historical context; the test that follows it is now a
positive regression test (closed-form OLS anchor) plus a companion warning
test, not an xfail.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.models.timeseries import _favar_minnesota_prior


# ---------------------------------------------------------------------------
# Anchor 1: Minnesota prior variance formula (deterministic).
# ---------------------------------------------------------------------------


def _hand_ar_sigma(values: np.ndarray, n_lag: int) -> np.ndarray:
    """Residual SD of an independently-written univariate AR(n_lag) OLS fit.

    Mirrors the documented ``sigma="AR"`` convention of
    ``bvartools::minnesota_prior``: regress each series on its own first
    ``n_lag`` lags (OLS, no intercept -- the panel entering
    ``bvar_minnesota``/``bvar_normal_inverse_wishart`` is demeaned before the
    prior is built) and take the residual standard deviation.
    """
    n_obs, k = values.shape
    sigmas = np.empty(k, dtype=float)
    for series in range(k):
        y = values[n_lag:, series]
        design = np.column_stack(
            [values[n_lag - lag : n_obs - lag, series] for lag in range(1, n_lag + 1)]
        )
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ coef
        dof = max(1, len(y) - design.shape[1])
        sigmas[series] = np.sqrt(float(resid @ resid) / dof)
    return sigmas


def _hand_minnesota_variance_grid(
    values: np.ndarray, n_lag: int, kappa0: float, kappa1: float
) -> np.ndarray:
    """Independent re-derivation of the documented Minnesota prior variance grid.

    Returns a ``(k, k*n_lag)`` array laid out as ``[eq, (lag-1)*k + reg]`` --
    the same layout ``_favar_minnesota_prior`` uses internally -- purely so
    the two can be compared element-wise; every VALUE is computed from the
    formula above, not copied from the source.
    """
    k = values.shape[1]
    sigma = _hand_ar_sigma(values, n_lag)
    variance = np.empty((k, k * n_lag), dtype=float)
    for lag in range(1, n_lag + 1):
        for eq in range(k):
            for reg in range(k):
                col = (lag - 1) * k + reg
                if eq == reg:
                    variance[eq, col] = kappa0 / lag**2
                else:
                    variance[eq, col] = (
                        kappa0 * kappa1 / lag**2 * sigma[eq] ** 2 / sigma[reg] ** 2
                    )
    return variance


def _correlated_var_panel(n_obs: int = 120, seed: int = 20260704) -> np.ndarray:
    rng = np.random.default_rng(seed)
    phi = np.array([[0.5, 0.1, -0.05], [0.05, 0.4, 0.1], [0.0, -0.1, 0.6]])
    values: np.ndarray = np.zeros((n_obs, 3), dtype=float)
    for t in range(1, n_obs):
        values[t] = phi @ values[t - 1] + rng.normal(scale=[1.0, 1.5, 0.7])
    return values


@pytest.mark.reference
@pytest.mark.parametrize("kappa0,kappa1,n_lag", [(2.0, 0.5, 2), (5.0, 0.2, 1), (1.0, 1.0, 3)])
def test_minnesota_prior_variance_matches_documented_formula(kappa0, kappa1, n_lag):
    values = _correlated_var_panel()
    _, precision = _favar_minnesota_prior(values, n_lag, kappa0, kappa1)
    expected_variance = _hand_minnesota_variance_grid(values, n_lag, kappa0, kappa1)
    expected_precision_diag = 1.0 / expected_variance.flatten(order="F")
    actual_precision_diag = np.diagonal(precision)
    np.testing.assert_allclose(
        actual_precision_diag, expected_precision_diag, rtol=0, atol=1e-12
    )


@pytest.mark.reference
def test_minnesota_prior_own_lag_ignores_cross_sectional_scale():
    """Own-lag variance kappa0/l^2 must not depend on which equation/lag-decay.

    A direct, exact-value check of the "shrinkage pattern" itself (own-lag vs
    cross-lag vs lag-decay) independent of the AR-sigma computation: own-lag
    precision entries must equal exactly lag**2 / kappa0 for every equation,
    and must strictly increase (tighten) with lag.
    """
    values = _correlated_var_panel()
    kappa0, kappa1, n_lag = 3.0, 0.4, 3
    _, precision = _favar_minnesota_prior(values, n_lag, kappa0, kappa1)
    k = values.shape[1]
    precision_grid = np.diagonal(precision).reshape((k * n_lag, k), order="C")
    # position of the own-lag entry for equation eq at lag `lag` is column
    # (lag-1)*k + eq (row eq of the `variance`/`precision` grid).
    for lag in range(1, n_lag + 1):
        expected_own = lag**2 / kappa0
        for eq in range(k):
            col = (lag - 1) * k + eq
            np.testing.assert_allclose(precision_grid[col, eq], expected_own, rtol=0, atol=1e-12)
    # Lag decay: precision must strictly increase (tighter shrinkage) with lag
    # for every (eq) own-lag term, since kappa0/l^2 strictly decreases in l.
    own_by_lag = np.array([lag**2 / kappa0 for lag in range(1, n_lag + 1)])
    assert np.all(np.diff(own_by_lag) > 0), own_by_lag


# ---------------------------------------------------------------------------
# Anchor 2: NIW Gibbs posterior mean vs closed-form OLS (Monte Carlo).
# ---------------------------------------------------------------------------


def _demeaned_var_design(panel: pd.DataFrame, n_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Independently-written lag-stacked VAR design on the demeaned panel.

    Reproduces the *contract* that ``_BayesianVAR.fit`` documents (demean the
    panel, then build a no-intercept VAR(n_lag) design with lag blocks in
    natural order 1..n_lag, each block in the panel's column order) without
    importing the estimator's own ``_var_design`` helper.
    """
    values = (panel - panel.mean()).to_numpy(dtype=float)
    n_obs, k = values.shape
    rows: list[list[float]] = []
    response: list[list[float]] = []
    for t in range(n_lag, n_obs):
        row: list[float] = []
        for lag in range(1, n_lag + 1):
            row.extend(values[t - lag].tolist())
        rows.append(row)
        response.append(values[t].tolist())
    return np.asarray(rows, dtype=float), np.asarray(response, dtype=float)


@pytest.mark.reference
def test_niw_gibbs_posterior_mean_matches_closed_form_ols():
    panel = pd.DataFrame(_correlated_var_panel(n_obs=120, seed=7), columns=["y1", "y2", "y3"])
    n_lag = 1

    design, response = _demeaned_var_design(panel, n_lag)
    # Closed-form target: with a diffuse prior on vec(A) (vb0=0), the GLS
    # estimator equals the row-wise OLS estimator on the SHARED design,
    # regardless of the (unknown, sampled) error covariance Sigma -- see
    # module docstring. Sanity: this is plain multivariate OLS.
    ols_coef, *_ = np.linalg.lstsq(design, response, rcond=None)  # (k*n_lag, k)

    fit = mf.models.bvar_normal_inverse_wishart(
        panel,
        n_lag=n_lag,
        b0=0.0,
        vb0=0.0,
        nu0=0.0,
        s0=1.0,
        iter=6000,
        burnin=1000,
        random_state=20260704,
    )
    gibbs_mean = fit.diagnostics["coef_mean"].to_numpy()  # (k, k*n_lag), eq rows
    gibbs_mcse = fit.diagnostics["coef_mcse"].to_numpy()

    # ols_coef is (k*n_lag regressors, k equations); transpose to (k eq, k*n_lag regressors)
    ols_coef_eq_major = ols_coef.T
    assert gibbs_mean.shape == ols_coef_eq_major.shape

    # Generous-but-justified MC tolerance: 6 x the sampler's own reported
    # Monte-Carlo standard error of the posterior mean, floored at a small
    # absolute value to avoid spurious failures on near-zero coefficients
    # whose MCSE also happens to be tiny.
    tolerance = np.maximum(6.0 * gibbs_mcse, 5e-3)
    diff = np.abs(gibbs_mean - ols_coef_eq_major)
    assert np.all(diff <= tolerance), (diff, tolerance, gibbs_mean, ols_coef_eq_major)


@pytest.mark.reference
def test_niw_gibbs_posterior_mean_matches_closed_form_ols_two_lags():
    """Same anchor at n_lag=2 to guard against a lag-count-specific bug."""
    panel = pd.DataFrame(_correlated_var_panel(n_obs=150, seed=11), columns=["a", "b", "c"])
    n_lag = 2

    design, response = _demeaned_var_design(panel, n_lag)
    ols_coef, *_ = np.linalg.lstsq(design, response, rcond=None)

    fit = mf.models.bvar_normal_inverse_wishart(
        panel,
        n_lag=n_lag,
        b0=0.0,
        vb0=0.0,
        nu0=0.0,
        s0=1.0,
        iter=6000,
        burnin=1000,
        random_state=20260704,
    )
    gibbs_mean = fit.diagnostics["coef_mean"].to_numpy()
    gibbs_mcse = fit.diagnostics["coef_mcse"].to_numpy()
    ols_coef_eq_major = ols_coef.T
    assert gibbs_mean.shape == ols_coef_eq_major.shape

    tolerance = np.maximum(6.0 * gibbs_mcse, 5e-3)
    diff = np.abs(gibbs_mean - ols_coef_eq_major)
    assert np.all(diff <= tolerance), (diff, tolerance, gibbs_mean, ols_coef_eq_major)


# ---------------------------------------------------------------------------
# FINDING (RESOLVED): bvar_normal_inverse_wishart's own default s0=0.0 (an
# exactly-zero inverse-Wishart prior scale) was numerically dangerous when
# the fitted VAR's residual covariance is even mildly near-singular -- a
# realistic, not pathological, situation (e.g. any near-collinear macro
# block, or a FAVAR-style factor+target system where the factors are
# DESIGNED to explain the target well). Discovered while building the
# ``favar`` noiseless-DGP oracle in ``test_favar_anchors.py``: favar()'s own
# default ``varprior=None`` silently resolved (``_parse_favar_varprior``
# treats the empty ``{}`` dict as falsy) to this exact same
# s0=0.0/vb0=0.0 configuration, and produced one-step forecasts many orders
# of magnitude off (observed -8.6e24, -881948, -24374 vs a true value of
# about -1.0 across DGP noise-level variants).
#
# Root cause isolated by direct parameter sweep on bvar_normal_inverse_wishart
# alone (bypassing FAVAR's factor/loading machinery entirely) on a fixed
# 3-variable VAR(1) panel where one variable is (almost) an exact linear
# combination of the other two:
#   s0=0.0   -> 97% of post-burn-in draws have some |coefficient| > 10;
#               reported posterior mean off by 4 orders of magnitude.
#   s0=1e-6  -> same failure mode (97%).
#   s0=1e-3  -> partial recovery (26% of draws still divergent).
#   s0=1.0   -> well-behaved, matches OLS/true values closely (5.5% divergent
#               draws, within normal MC noise).
# This isolated the mechanism to the interaction between an (exactly or
# near-)zero inverse-Wishart prior scale and a near-singular residual
# covariance in the Wishart-draw step of ``_favar_bvar_draws`` -- NOT
# something specific to FAVAR's standardization or factor-extraction code.
# It was not merely rare-draw noise: the MEDIAN draw was also badly wrong, so
# no MCSE-based tolerance could have papered over it.
#
# FIX: ``bvar_normal_inverse_wishart``/``bvar_minnesota``'s ``s0`` default is
# now ``None``, which `_favar_bvar_draws` resolves to a data-dependent
# diagonal scale ``diag(sigma_1**2, ..., sigma_k**2)`` (per-equation
# AR(n_lag)-OLS residual variance -- see `_favar_default_niw_scale`), instead
# of the literal-zero scale. This keeps the Wishart draw's scale matrix
# strictly positive-definite regardless of how (near-)singular the sample
# residual covariance is. A companion divergence guard
# (`_warn_if_bvar_draws_diverged`) additionally fires a `UserWarning` if
# post-burn-in draws ever do explode (e.g. an explicitly passed `s0=0.0`),
# so that path is no longer silent either. Two tests now cover this: a
# positive regression test that the default-s0 posterior mean matches the
# closed-form OLS anchor on this exact previously-diverging fixture, and a
# companion test that an explicit `s0=0.0` still reproduces the old
# divergence AND raises the new warning.
# ---------------------------------------------------------------------------


def _near_singular_favar_like_panel(n_obs: int = 200, seed: int = 2026) -> pd.DataFrame:
    """The exact fixture that originally triggered the WP-V2 finding.

    A 2x2 rotation embedded in a 3-variable VAR(1) system where the third
    variable ("y") is (almost) an exact linear combination of the other two
    ("f1", "f2") -- i.e. a near-singular residual covariance, the realistic
    FAVAR-style scenario (a target well-explained by its own factors) that
    the finding's root-cause sweep isolated.
    """
    rng = np.random.default_rng(seed)
    a = np.array(
        [[np.cos(np.pi / 9), -np.sin(np.pi / 9)], [np.sin(np.pi / 9), np.cos(np.pi / 9)]]
    )
    state: np.ndarray = np.zeros((n_obs, 2), dtype=float)
    state[0] = [1.0, 0.4]
    for t in range(1, n_obs):
        state[t] = a @ state[t - 1]
    target = state @ np.array([0.8, -0.6]) + 1e-3 * rng.normal(size=n_obs)
    return pd.DataFrame(
        np.column_stack([state, target]),
        columns=["f1", "f2", "y"],
        index=pd.date_range("1990-01-31", periods=n_obs, freq="ME"),
    )


@pytest.mark.reference
def test_niw_default_s0_matches_closed_form_ols_on_previously_diverging_fixture():
    """Regression test for the (now fixed) WP-V2 finding.

    Same closed-form-OLS anchor as
    ``test_niw_gibbs_posterior_mean_matches_closed_form_ols`` above (the
    flat coefficient prior b0=vb0=0 makes the Gibbs sampler's coefficient
    posterior mean equal plain OLS on the shared VAR design for ANY
    positive-definite Sigma draw -- unaffected by what s0 is), but run on
    the exact near-singular fixture that used to diverge under the old
    s0=0.0 default. With the new data-dependent default, this must now pass
    at the same MCSE-based tolerance as the other OLS-anchor tests, not just
    "less than some generous bound".
    """
    panel = _near_singular_favar_like_panel()
    n_lag = 1
    design, response = _demeaned_var_design(panel, n_lag)
    ols_coef, *_ = np.linalg.lstsq(design, response, rcond=None)
    ols_coef_eq_major = ols_coef.T

    fit = mf.models.bvar_normal_inverse_wishart(
        panel, n_lag=n_lag, iter=6000, burnin=1000, random_state=2026
    )  # b0/vb0/nu0/s0 all left at their documented defaults (s0=None now).
    gibbs_mean = fit.diagnostics["coef_mean"].to_numpy()
    gibbs_mcse = fit.diagnostics["coef_mcse"].to_numpy()
    assert gibbs_mean.shape == ols_coef_eq_major.shape

    tolerance = np.maximum(6.0 * gibbs_mcse, 5e-3)
    diff = np.abs(gibbs_mean - ols_coef_eq_major)
    assert np.all(diff <= tolerance), (diff, tolerance, gibbs_mean, ols_coef_eq_major)

    # True coefficients (the embedded 2x2 rotation, plus a near-zero third
    # row) are all in [-1, 1]; a well-behaved posterior mean must be too.
    assert np.max(np.abs(gibbs_mean)) < 5.0, gibbs_mean


@pytest.mark.reference
def test_niw_explicit_s0_zero_still_diverges_and_warns():
    """An explicitly passed s0=0.0 keeps its old (diverging) behavior exactly,
    but is no longer silent: `_warn_if_bvar_draws_diverged` must now raise a
    `UserWarning` naming the cause. This is the regression test for the
    divergence guard itself, and confirms explicit callers are unaffected by
    the new *default* (i.e. the fix only changes what happens when s0 is
    left unspecified).
    """
    panel = _near_singular_favar_like_panel()
    with pytest.warns(UserWarning, match="near-singular residual covariance"):
        fit = mf.models.bvar_normal_inverse_wishart(
            panel, n_lag=1, iter=4000, burnin=1000, random_state=2026, s0=0.0
        )
    coef_mean = fit.diagnostics["coef_mean"].to_numpy()
    # Confirms this is genuinely the same old failure mode reproduced (not a
    # spurious warning on well-behaved output): the posterior mean must still
    # be wildly off from the true [-1, 1]-bounded coefficients.
    assert np.max(np.abs(coef_mean)) > 10.0, coef_mean
