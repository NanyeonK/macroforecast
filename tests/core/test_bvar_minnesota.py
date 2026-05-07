"""Issues #185 / #186 -- BVAR Minnesota and Normal-Inverse-Wishart prior
estimators are operational.

The closed-form posterior mean is

    β̂ = (V⁻¹ + X'X)⁻¹ (V⁻¹ m + X'y)

with ``m`` placing unit weight on the first own-lag column when present
and ``V`` shrinking higher lags via the Litterman (1986) decay scheme.

Pins:

* Both families pass the L4 validator (operational status).
* ``_BayesianVAR.fit`` produces a closed-form posterior mean -- not the
  plain VAR coefficient -- by checking that strong-prior settings pull
  predictions toward the random-walk forecast.
* NIW differs from Minnesota in the σ² hyperparameter (heavier tails)
  -- their predictions diverge under the same data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from macroforecast.core.runtime import _BayesianVAR
from macroforecast.core.ops.l4_ops import (
    OPERATIONAL_MODEL_FAMILIES,
    FUTURE_MODEL_FAMILIES,
    get_family_status,
)


def test_bvar_minnesota_is_operational():
    assert "bvar_minnesota" in OPERATIONAL_MODEL_FAMILIES
    assert "bvar_minnesota" not in FUTURE_MODEL_FAMILIES
    assert get_family_status("bvar_minnesota") == "operational"


def test_bvar_normal_inverse_wishart_is_operational():
    assert "bvar_normal_inverse_wishart" in OPERATIONAL_MODEL_FAMILIES
    assert "bvar_normal_inverse_wishart" not in FUTURE_MODEL_FAMILIES
    assert get_family_status("bvar_normal_inverse_wishart") == "operational"


def _toy_data(n: int = 60, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    y = pd.Series(rng.normal(size=n).cumsum(), name="y")
    X = pd.DataFrame(
        {
            "y_lag1": y.shift(1),
            "x1_lag1": rng.normal(size=n),
            "x1_lag2": rng.normal(size=n),
        }
    )
    return X, y


def test_minnesota_random_walk_anchor_pulls_lag1_coef_toward_one():
    """Strong-prior limit (very small λ₁ → tight prior) should pull the
    own-lag-1 coefficient toward the random-walk anchor m=1."""

    X, y = _toy_data()
    loose = _BayesianVAR(prior="bvar_minnesota", lambda1=10.0).fit(X, y)
    tight = _BayesianVAR(prior="bvar_minnesota", lambda1=1e-3).fit(X, y)
    # ``y_lag1`` is the first column; the random-walk anchor sits there.
    assert loose._coef is not None and tight._coef is not None
    # As the prior tightens, the coefficient on y_lag1 must move toward 1.
    assert abs(tight._coef[0] - 1.0) < abs(loose._coef[0] - 1.0)


def test_minnesota_predicts_finite_values():
    X, y = _toy_data()
    model = _BayesianVAR(prior="bvar_minnesota").fit(X, y)
    preds = model.predict(X.fillna(0.0))
    assert preds.shape == (len(X),)
    assert np.all(np.isfinite(preds))


def test_niw_differs_from_minnesota_under_same_data():
    """Normal-Inverse-Wishart bumps λ₁ by the documented factor; the
    posterior mean coefficients (and therefore predictions) must differ
    from the plain Minnesota fit."""

    X, y = _toy_data()
    a = _BayesianVAR(prior="bvar_minnesota", lambda1=0.2).fit(X, y)
    b = _BayesianVAR(prior="bvar_normal_inverse_wishart", lambda1=0.2).fit(X, y)
    # Predictions should differ because λ₁ scaling is different.
    pa = a.predict(X.fillna(0.0))
    pb = b.predict(X.fillna(0.0))
    assert not np.allclose(pa, pb)


def test_bvar_handles_missing_columns_at_predict_time():
    X, y = _toy_data()
    model = _BayesianVAR(prior="bvar_minnesota").fit(X, y)
    # Predict with a frame missing one column -- should silently zero it.
    X_partial = X.drop(columns=["x1_lag2"]).fillna(0.0)
    preds = model.predict(X_partial)
    assert preds.shape == (len(X_partial),)
    assert np.all(np.isfinite(preds))


def test_bvar_posterior_irf_populated_by_default():
    """v0.9.0a0 audit-fix: ``n_draws`` default is now 500 (was 0). A
    fitted BVAR carries the posterior IRF dict so L7 ops can route the
    Bayesian path. Without this default the posterior IRF was orphaned
    and L7 fell back to the OLS-VAR ``.irf()`` builder."""

    X, y = _toy_data(n=120, seed=2)
    model = _BayesianVAR(prior="bvar_minnesota", n_draws=120, posterior_irf_periods=8).fit(X, y)
    assert model._posterior_irf is not None
    assert {"mean", "p16", "p84"}.issubset(set(model._posterior_irf.keys()))
    # Shape: (n_periods+1, K, K) where K is the endog count.
    mean_irf = np.asarray(model._posterior_irf["mean"])
    assert mean_irf.ndim == 3 and mean_irf.shape[0] == 9


def test_bvar_results_is_multi_equation_minnesota_not_ols_var():
    """Audit gap-fix #5b: ``_results`` is now a multi-equation Minnesota
    BVAR (one Minnesota-prior posterior per endogenous variable) rather
    than statsmodels' OLS VAR. Pin the canonical attributes and verify
    the params shape is the K-equation joint coefficient matrix."""

    X, y = _toy_data(n=120, seed=4)
    model = _BayesianVAR(prior="bvar_minnesota", n_draws=0).fit(X, y)
    assert model._results is not None
    # Joint panel: __y__ + 3 features = 4 endogenous variables.
    assert model._results.endog.shape[1] == 4
    # params shape: (1 + K·p, K) = (1 + 4·2, 4) = (9, 4)
    assert model._results.params.shape == (9, 4)
    # Names include the target __y__ first.
    assert model._results.names[0] == "__y__"
    # Sigma_u is K×K and symmetric.
    K = model._results.endog.shape[1]
    assert model._results.sigma_u.shape == (K, K)
    np.testing.assert_allclose(model._results.sigma_u, model._results.sigma_u.T, atol=1e-8)


def test_bvar_minnesota_random_walk_anchor_applies_to_every_equation():
    """Per-equation Minnesota: each variable's own-lag-1 coefficient
    must be pulled toward 1.0 by a tight prior. This is the multi-
    equation generalisation of the existing single-equation random-walk
    anchor pin."""

    X, y = _toy_data(n=120, seed=5)
    loose = _BayesianVAR(prior="bvar_minnesota", lambda1=10.0, n_draws=0).fit(X, y)
    tight = _BayesianVAR(prior="bvar_minnesota", lambda1=1e-3, n_draws=0).fit(X, y)
    K = loose._results.endog.shape[1]
    # Own-lag-1 coefficient for variable i is at B[i, 1 + 0·K + i] = B[i, 1+i].
    for i in range(K):
        idx = 1 + i
        loose_own = loose._results._B[i, idx]
        tight_own = tight._results._B[i, idx]
        # Tight prior should be closer to 1.0 than loose for every variable.
        assert abs(tight_own - 1.0) <= abs(loose_own - 1.0) + 1e-6, (
            f"variable {i}: tight {tight_own:.3f} not closer to 1 than loose {loose_own:.3f}"
        )


def test_bvar_irf_builder_orth_irfs_zero_horizon_equals_cholesky():
    """``_MultiEquationBVARResults.irf(n).orth_irfs[0]`` is the Cholesky
    factor of Σ_u (Sims 1980 IRF construction)."""

    X, y = _toy_data(n=120, seed=6)
    model = _BayesianVAR(prior="bvar_minnesota", n_draws=0).fit(X, y)
    irfs = model._results.irf(8).orth_irfs
    K = model._results.endog.shape[1]
    chol = np.linalg.cholesky(model._results.sigma_u + 1e-10 * np.eye(K))
    np.testing.assert_allclose(irfs[0], chol, atol=1e-9)
    # Shape invariant.
    assert irfs.shape == (9, K, K)


def test_bvar_fevd_shares_sum_to_one_per_horizon_per_response():
    """FEVD invariant: at every horizon, each response variable's shares
    across shocks sum to 1.0 (standard FEVD identity)."""

    X, y = _toy_data(n=120, seed=7)
    model = _BayesianVAR(prior="bvar_minnesota", n_draws=0).fit(X, y)
    decomp = model._results.fevd(12).decomp
    K = model._results.endog.shape[1]
    assert decomp.shape == (12, K, K)
    sums = decomp.sum(axis=2)
    np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-6)


def test_bvar_posterior_irf_uses_multivariate_minnesota_covariance():
    """Audit gap-fix #14: posterior IRF sampling uses the multi-equation
    Minnesota block-diagonal covariance instead of the OLS asymptotic
    flat-prior approximation. Verify the credible-band width is non-zero
    (sampling is happening) and that sigma2-scaled posterior covariance
    is cached on results.posterior_cov_per_eq."""

    X, y = _toy_data(n=120, seed=8)
    model = _BayesianVAR(
        prior="bvar_minnesota", n_draws=80, posterior_irf_periods=8,
    ).fit(X, y)
    assert model._results is not None
    assert model._results.posterior_cov_per_eq is not None
    K = model._results.endog.shape[1]
    n_coef = 1 + K * model._results.k_ar
    assert model._results.posterior_cov_per_eq.shape == (K, n_coef, n_coef)
    # Posterior covariance must be symmetric per-equation.
    for i in range(K):
        cov_i = model._results.posterior_cov_per_eq[i]
        np.testing.assert_allclose(cov_i, cov_i.T, atol=1e-8)
    # IRF bands have non-trivial width (sampling actually happened).
    p16 = np.asarray(model._posterior_irf["p16"])
    p84 = np.asarray(model._posterior_irf["p84"])
    assert np.any(np.abs(p84 - p16) > 1e-6)


def test_bvar_posterior_fevd_bands_surface_in_l7_frame():
    """Audit gap-fix #14: FEVD posterior bands now surface as p16/p84
    columns on the L7 frame when ``_posterior_irf`` carries
    ``fevd_mean`` etc."""

    from macroforecast.core.runtime import _var_impulse_frame
    from macroforecast.core.types import ModelArtifact

    X, y = _toy_data(n=120, seed=9)
    model = _BayesianVAR(
        prior="bvar_minnesota", n_draws=80, posterior_irf_periods=8,
    ).fit(X, y)
    assert "fevd_mean" in model._posterior_irf
    artifact = ModelArtifact(
        model_id="bvar_fevd_test", family="bvar_minnesota",
        fitted_object=model, framework="numpy",
        feature_names=tuple(X.columns),
    )
    frame = _var_impulse_frame(artifact, op_name="fevd", n_periods=8)
    assert (frame["status"] == "posterior_mean").all()
    assert {"p16", "p84"}.issubset(set(frame.columns))
    # Variance shares are non-negative.
    assert (frame["importance"] >= -1e-9).all()
    assert (frame["p16"] >= -1e-9).all()
    assert (frame["p84"] >= -1e-9).all()


def test_bvar_l7_irf_routes_through_posterior_when_available():
    """``_var_impulse_frame`` consults ``_posterior_irf`` first; status
    column reflects ``posterior_mean`` and credible-band columns p16/p84
    surface (Coulombe & Göbel 2021 §3 reporting convention)."""

    from macroforecast.core.runtime import _var_impulse_frame
    from macroforecast.core.types import ModelArtifact

    X, y = _toy_data(n=120, seed=3)
    model = _BayesianVAR(prior="bvar_minnesota", n_draws=120, posterior_irf_periods=8).fit(X, y)
    artifact = ModelArtifact(
        model_id="bvar_test",
        family="bvar_minnesota",
        fitted_object=model,
        framework="numpy",
        feature_names=tuple(X.columns),
    )
    frame = _var_impulse_frame(artifact, op_name="orthogonalised_irf", n_periods=8)
    assert (frame["status"] == "posterior_mean").all()
    assert {"p16", "p84"}.issubset(set(frame.columns))
    assert np.all(np.isfinite(frame["p16"].values))
    assert np.all(np.isfinite(frame["p84"].values))
