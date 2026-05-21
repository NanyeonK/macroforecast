"""Independent tester validation for C49 realized_garch promotion.

Produced by the StatsClaw tester pipeline (Cycle 49). Tests are derived
exclusively from test-spec.md (Cycle 49, realized_garch). The tester does NOT
read spec.md or implementation.md — results are validated purely against
behavioral contracts.

Coverage:
  R   — Registration (family in OPERATIONAL, not FUTURE)
  C   — Contract/shape (fit-predict shape, dtype, finiteness)
  REC — Parameter recovery (Hansen-Huang-Shek synthetic DGP, T=500)
  SD  — Seed determinism (same seed -> bit-identical predictions)
  EC  — Edge cases (too few obs, missing RV, NaN returns, zero-variance RV)
  REG — Regression guards (realized_garch_with_rv_exog still works; produces
        distinct variance forecasts from the new joint-MLE variant)
  XR  — Cross-reference (arch GARCH(1,1) correlation sanity; skipif no arch)

References:
  Hansen, P.R., Huang, Z. & Shek, H.H. (2012) "Realized GARCH: a joint model
  for returns and realized measures of volatility." Journal of Applied
  Econometrics 27(6): 877-906.

Note on _build_l4_model interface:
  The function signature is _build_l4_model(family, params) where params is a
  dict; the seed is conveyed via params["random_state"] (not as a keyword arg).
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from macroforecast.core.runtime import _build_l4_model
from macroforecast.core.ops.l4_ops import (
    OPERATIONAL_MODEL_FAMILIES,
    FUTURE_MODEL_FAMILIES,
    get_family_status,
)


# ---------------------------------------------------------------------------
# DGP helper — Hansen-Huang-Shek (2012) joint process (test-spec.md §2.1)
# ---------------------------------------------------------------------------

def _simulate_realized_garch_dgp(
    T: int = 500, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series, dict[str, float]]:
    """Simulate Hansen-Huang-Shek (2012) joint process.

    True parameters:
        mu = 0.05, omega = -0.5, beta = 0.85,
        tau_1 = -0.10, tau_2 = 0.05, gamma = 0.05,
        xi = -0.2, phi = 0.90, delta_1 = -0.05, delta_2 = 0.05,
        sigma_u = 0.15

    Returns (X, y, true_params):
        y : pd.Series of returns r_t, length T
        X : pd.DataFrame with column "rv" = realized variance x_t, length T
        true_params : dict of true parameter values
    """
    rng = np.random.default_rng(seed)
    true_params: dict[str, float] = {
        "mu": 0.05, "omega": -0.5, "beta": 0.85,
        "tau_1": -0.10, "tau_2": 0.05, "gamma": 0.05,
        "xi": -0.2, "phi": 0.90,
        "delta_1": -0.05, "delta_2": 0.05,
        "sigma_u": 0.15,
    }
    h = np.zeros(T)
    z = np.zeros(T)
    u = np.zeros(T)
    r = np.zeros(T)
    x = np.zeros(T)
    # Stationary initialisation: unconditional h = exp(omega / (1 - beta))
    h[0] = float(np.exp(true_params["omega"] / (1.0 - true_params["beta"])))
    for t in range(T):
        z[t] = float(rng.standard_normal())
        r[t] = true_params["mu"] + math.sqrt(h[t]) * z[t]
        u[t] = true_params["sigma_u"] * float(rng.standard_normal())
        log_x_t = (
            true_params["xi"]
            + true_params["phi"] * math.log(h[t])
            + true_params["delta_1"] * z[t]
            + true_params["delta_2"] * (z[t] ** 2 - 1.0)
            + u[t]
        )
        x[t] = float(np.exp(log_x_t))
        if t < T - 1:
            log_h_next = (
                true_params["omega"]
                + true_params["beta"] * math.log(h[t])
                + true_params["tau_1"] * z[t]
                + true_params["tau_2"] * (z[t] ** 2 - 1.0)
                + true_params["gamma"] * u[t]
            )
            h[t + 1] = float(np.exp(log_h_next))

    idx = pd.RangeIndex(T)
    X = pd.DataFrame({"rv": x}, index=idx)
    y = pd.Series(r, index=idx, name="returns")
    return X, y, true_params


def _build_rg(params: dict | None = None, random_state: int = 0):
    """Helper: build a realized_garch model via _build_l4_model.

    Seed is passed via params['random_state'] (not as a keyword argument).
    """
    p = {"random_state": random_state}
    if params:
        p.update(params)
    return _build_l4_model("realized_garch", params=p)


def _build_rv_exog(params: dict | None = None, random_state: int = 0):
    """Helper: build a realized_garch_with_rv_exog model."""
    p = {"random_state": random_state}
    if params:
        p.update(params)
    return _build_l4_model("realized_garch_with_rv_exog", params=p)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dgp_small() -> tuple[pd.DataFrame, pd.Series]:
    """T=100, seed=0 — contract / shape tests."""
    X, y, _ = _simulate_realized_garch_dgp(T=100, seed=0)
    return X, y


@pytest.fixture
def dgp_medium() -> tuple[pd.DataFrame, pd.Series]:
    """T=80, seed=0 — determinism and regression guard tests."""
    X, y, _ = _simulate_realized_garch_dgp(T=80, seed=0)
    return X, y


# ---------------------------------------------------------------------------
# Section R — Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    """BC-RG-1 to BC-RG-3: family registration and basic construction."""

    def test_c49_realized_garch_in_operational(self) -> None:
        """R-1: realized_garch must be in OPERATIONAL_MODEL_FAMILIES after C49."""
        assert "realized_garch" in OPERATIONAL_MODEL_FAMILIES, (
            "realized_garch must be OPERATIONAL after C49 promotion"
        )

    def test_c49_realized_garch_not_in_future(self) -> None:
        """R-2: realized_garch must NOT be in FUTURE_MODEL_FAMILIES after C49.
        After C49 FUTURE_MODEL_FAMILIES is empty.
        """
        assert "realized_garch" not in FUTURE_MODEL_FAMILIES, (
            "realized_garch must NOT be FUTURE after C49 promotion"
        )
        assert len(FUTURE_MODEL_FAMILIES) == 0, (
            f"FUTURE_MODEL_FAMILIES should be empty after C49; got {FUTURE_MODEL_FAMILIES}"
        )

    def test_c49_get_family_status_returns_operational(self) -> None:
        """R-3: get_family_status('realized_garch') == 'operational'."""
        assert get_family_status("realized_garch") == "operational"

    def test_c49_build_l4_model_no_exception(self) -> None:
        """R-4: _build_l4_model('realized_garch', ...) returns a fittable object
        without raising (BC-RG-3). seed is passed via params['random_state'].
        """
        model = _build_rg(random_state=0)
        assert model is not None
        assert callable(getattr(model, "fit", None)), "model must have .fit()"
        assert callable(getattr(model, "predict", None)), "model must have .predict()"


# ---------------------------------------------------------------------------
# Section C — Contract / Shape
# ---------------------------------------------------------------------------

class TestContractShape:
    """BC-RG-4 to BC-RG-6, BC-RG-10: output shape/dtype/finiteness."""

    def test_realized_garch_fit_predict_shape(
        self, dgp_small: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """C-1: predict(X) returns array of shape (T,) with finite float values
        (BC-RG-4, test-spec.md §2.3).
        """
        X, y = dgp_small
        model = _build_rg(params={"realized_variance": "rv"}, random_state=0)
        model.fit(X, y)
        pred = model.predict(X)
        assert isinstance(pred, np.ndarray), "predict() must return np.ndarray"
        assert pred.shape == (len(X),), f"Expected shape ({len(X)},), got {pred.shape}"
        assert np.all(np.isfinite(pred)), "predict() values must all be finite"

    def test_realized_garch_predict_variance_shape_and_positivity(
        self, dgp_small: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """C-2: predict_variance(h_steps=5) returns shape (5,) with positive
        finite values (BC-RG-5, test-spec.md §2.3).
        """
        X, y = dgp_small
        model = _build_rg(params={"realized_variance": "rv"}, random_state=0)
        model.fit(X, y)
        var_fc = model.predict_variance(h_steps=5)
        assert isinstance(var_fc, np.ndarray), "predict_variance() must return np.ndarray"
        assert var_fc.shape == (5,), f"Expected shape (5,), got {var_fc.shape}"
        assert np.all(np.isfinite(var_fc)), "predict_variance() values must be finite"
        assert np.all(var_fc > 0), "predict_variance() values must be positive"

    def test_realized_garch_conditional_volatility_shape(
        self, dgp_small: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """C-3: conditional_volatility_ is an array of length <= T with positive
        finite values after fitting (BC-RG-10, test-spec.md §2.3).
        """
        X, y = dgp_small
        model = _build_rg(params={"realized_variance": "rv"}, random_state=0)
        model.fit(X, y)
        cv = model.conditional_volatility_
        assert cv is not None, "conditional_volatility_ must not be None after fitting"
        assert isinstance(cv, np.ndarray), "conditional_volatility_ must be np.ndarray"
        assert len(cv) <= len(X), f"cv length {len(cv)} > T={len(X)}"
        assert len(cv) >= 30, "cv must have at least min-obs rows"
        assert np.all(np.isfinite(cv)), "conditional_volatility_ values must be finite"
        assert np.all(cv > 0), "conditional_volatility_ must be positive"

    def test_realized_garch_params_dict_contains_required_keys(
        self, dgp_small: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """C-4: params_ contains at minimum mu, omega, beta, xi, phi (BC-RG-6,
        test-spec.md §2.3).
        """
        X, y = dgp_small
        model = _build_rg(params={"realized_variance": "rv"}, random_state=0)
        model.fit(X, y)
        p = model.params_
        assert isinstance(p, dict), "params_ must be a dict"
        assert len(p) > 0, "params_ must be non-empty after fitting"
        for key in ("mu", "omega", "beta", "xi", "phi"):
            assert key in p, f"params_ missing required key: {key!r}"
        for k, v in p.items():
            assert isinstance(v, (float, int, np.floating, np.integer)), (
                f"params_[{k!r}] is not numeric: {type(v)}"
            )
            assert math.isfinite(float(v)), f"params_[{k!r}] = {v} is not finite"

    def test_conditional_volatility_none_before_fit(self) -> None:
        """C-5: conditional_volatility_ is None before fit() is called (BC-RG-10)."""
        model = _build_rg(random_state=0)
        assert model.conditional_volatility_ is None


# ---------------------------------------------------------------------------
# Section REC — Parameter Recovery (T=500, seed=42)
# ---------------------------------------------------------------------------

class TestParameterRecovery:
    """Hansen-Huang-Shek (2012) synthetic DGP recovery (test-spec.md §2.2)."""

    def test_realized_garch_parameter_recovery_t500(self) -> None:
        """REC-1: Joint MLE recovers key parameters within tolerances.

        True params: mu=0.05, beta=0.85, phi=0.90, omega=-0.5.
        Tolerances (from test-spec.md §2.2):
          mu   within 0.05 of 0.05  -> abs(est - 0.05) <= 0.05
          beta within 0.15 of 0.85  -> est in [0.70, 1.00]
          phi  within 0.20 of 0.90  -> est in [0.70, 1.10]
          omega within 0.50 of -0.5 -> abs(est - (-0.5)) <= 0.50
        """
        X, y, true_params = _simulate_realized_garch_dgp(T=500, seed=42)
        model = _build_rg(
            params={"realized_variance": "rv"},
            random_state=42,
        )
        model.fit(X, y)
        p = model.params_

        # mu recovery: atol=0.05
        mu_true = true_params["mu"]   # 0.05
        mu_est = float(p["mu"])
        assert abs(mu_est - mu_true) <= 0.05, (
            f"mu recovery failed: true={mu_true}, est={mu_est}, "
            f"abs_error={abs(mu_est - mu_true):.4f} > atol=0.05"
        )

        # beta recovery: [0.70, 1.00]
        beta_est = float(p["beta"])
        assert 0.70 <= beta_est <= 1.00, (
            f"beta recovery failed: true=0.85, est={beta_est:.4f}, "
            "expected in [0.70, 1.00]"
        )

        # phi recovery: [0.70, 1.10]
        phi_est = float(p["phi"])
        assert 0.70 <= phi_est <= 1.10, (
            f"phi recovery failed: true=0.90, est={phi_est:.4f}, "
            "expected in [0.70, 1.10]"
        )

        # omega recovery: atol=0.50
        omega_true = true_params["omega"]  # -0.5
        omega_est = float(p["omega"])
        assert abs(omega_est - omega_true) <= 0.50, (
            f"omega recovery failed: true={omega_true}, est={omega_est:.4f}, "
            f"abs_error={abs(omega_est - omega_true):.4f} > atol=0.50"
        )


# ---------------------------------------------------------------------------
# Section SD — Seed Determinism
# ---------------------------------------------------------------------------

class TestSeedDeterminism:
    """BC-RG-7: same seed -> bit-identical predictions (test-spec.md §2.4)."""

    def test_realized_garch_seed_determinism(self) -> None:
        """SD-1: Two fits with the same random_state produce bit-identical
        predict() output (test-spec.md §2.4, tolerance: np.array_equal).
        """
        X, y, _ = _simulate_realized_garch_dgp(T=80, seed=99)

        model_a = _build_rg(
            params={"realized_variance": "rv"},
            random_state=7,
        )
        model_a.fit(X, y)
        pred_a = model_a.predict(X)

        model_b = _build_rg(
            params={"realized_variance": "rv"},
            random_state=7,
        )
        model_b.fit(X, y)
        pred_b = model_b.predict(X)

        assert np.array_equal(pred_a, pred_b), (
            "Seed determinism violated: predict() output differs between two fits "
            "with the same seed. "
            f"Max diff = {np.max(np.abs(pred_a - pred_b)):.3e}"
        )


# ---------------------------------------------------------------------------
# Section EC — Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """EC-RG-1 to EC-RG-5 (test-spec.md §4)."""

    def test_realized_garch_too_few_observations(self) -> None:
        """EC-RG-1: T < 30 raises NotImplementedError mentioning minimum T
        (test-spec.md §4 EC-RG-1 + §2.5).
        """
        X, y, _ = _simulate_realized_garch_dgp(T=15, seed=0)
        model = _build_rg(params={"realized_variance": "rv"}, random_state=0)
        with pytest.raises((NotImplementedError, ValueError)):
            model.fit(X, y)

    def test_realized_garch_missing_rv_column_fallback(self) -> None:
        """EC-RG-2: Missing 'rv' column triggers r^2 fallback; no exception
        (test-spec.md §2.6, EC-RG-2).
        """
        rng = np.random.default_rng(5)
        T = 100
        X_no_rv = pd.DataFrame({"x1": rng.standard_normal(T)})
        y = pd.Series(rng.standard_normal(T), name="returns")

        model = _build_rg(
            params={"realized_variance": "rv"},  # column is missing from X
            random_state=0,
        )
        model.fit(X_no_rv, y)
        pred = model.predict(X_no_rv)
        assert isinstance(pred, np.ndarray)
        assert pred.shape == (T,)
        assert np.all(np.isfinite(pred))

    def test_realized_garch_nan_in_returns(self) -> None:
        """EC-RG-3: NaN values in y are dropped before fitting; fit proceeds
        if >= 30 non-NaN observations remain (test-spec.md §4 EC-RG-3).
        """
        X, y, _ = _simulate_realized_garch_dgp(T=100, seed=0)
        y_with_nan = y.copy()
        nan_idx = np.arange(0, 100, 5)  # 20 positions
        y_with_nan.iloc[nan_idx] = np.nan

        model = _build_rg(params={"realized_variance": "rv"}, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y_with_nan)

        pred = model.predict(X)
        assert np.all(np.isfinite(pred))

    def test_realized_garch_constant_rv_no_crash(self) -> None:
        """EC-RG-4: Constant (zero-variance) RV column — fit completes without
        crashing (test-spec.md §4 EC-RG-4).
        """
        rng = np.random.default_rng(0)
        T = 50
        X_const = pd.DataFrame({"rv": np.ones(T) * 0.01})
        y = pd.Series(rng.standard_normal(T) * 0.01, name="returns")

        model = _build_rg(params={"realized_variance": "rv"}, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_const, y)


# ---------------------------------------------------------------------------
# Section REG — Regression Guards
# ---------------------------------------------------------------------------

def _arch_available() -> bool:
    try:
        import arch  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _arch_available(), reason="arch package not installed; realized_garch_with_rv_exog requires arch")
class TestRegressionGuards:
    """BC-RG-8: realized_garch_with_rv_exog stays operational (test-spec.md §2.7)
    and §9: the two variants produce distinct variance forecasts.
    Note: realized_garch_with_rv_exog depends on the arch package (pre-existing
    constraint, unchanged by C49). Tests are skipped when arch is unavailable.
    """

    def test_realized_garch_with_rv_exog_still_operational(
        self, dgp_medium: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """REG-1: realized_garch_with_rv_exog is still operational and produces
        correct shapes after the internal variant rename (BC-RG-8, test-spec.md §2.7).
        """
        X, y = dgp_medium
        model = _build_rv_exog(params={"realized_variance": "rv"}, random_state=0)
        model.fit(X, y)
        pred = model.predict(X)
        assert isinstance(pred, np.ndarray)
        assert pred.shape == (len(X),)
        assert np.all(np.isfinite(pred))

        cv = model.conditional_volatility_
        assert cv is not None, "realized_garch_with_rv_exog: conditional_volatility_ is None"
        assert np.all(np.isfinite(cv))

    def test_realized_garch_vs_rv_exog_produce_different_variance(
        self, dgp_medium: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """REG-2: The joint-MLE variant (realized_garch) produces DIFFERENT
        predict_variance output from the RV-exog approximation variant
        (realized_garch_with_rv_exog), confirming distinct algorithms
        (test-spec.md §9).
        """
        X, y = dgp_medium
        h_steps = 3

        model_joint = _build_rg(params={"realized_variance": "rv"}, random_state=0)
        model_joint.fit(X, y)
        var_joint = model_joint.predict_variance(h_steps=h_steps)

        model_exog = _build_rv_exog(params={"realized_variance": "rv"}, random_state=0)
        model_exog.fit(X, y)
        var_exog = model_exog.predict_variance(h_steps=h_steps)

        assert not np.allclose(var_joint, var_exog, rtol=1e-3), (
            "realized_garch and realized_garch_with_rv_exog produced identical "
            "predict_variance() output — expected distinct algorithms to differ. "
            f"joint={var_joint}, exog={var_exog}"
        )


# ---------------------------------------------------------------------------
# Section XR — Cross-Reference (arch GARCH(1,1), skip if arch unavailable)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _arch_available(), reason="arch package not installed")
class TestCrossReferenceArch:
    """XR: arch GARCH(1,1) correlation sanity (test-spec.md §2.8, §9)."""

    def test_realized_garch_vs_garch11_correlation_sanity(self) -> None:
        """XR-1: conditional_volatility_ of realized_garch is correlated (>0.5)
        with GARCH(1,1) conditional std from the arch package.
        """
        import arch as _arch  # type: ignore

        X, y, _ = _simulate_realized_garch_dgp(T=300, seed=0)

        model_rg = _build_rg(params={"realized_variance": "rv"}, random_state=0)
        model_rg.fit(X, y)
        cv_rg = model_rg.conditional_volatility_

        am = _arch.arch_model(y.values, vol="Garch", p=1, q=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = am.fit(disp="off")
        cv_arch = res.conditional_volatility

        n = min(len(cv_rg), len(cv_arch))
        corr = np.corrcoef(cv_rg[-n:], cv_arch[-n:])[0, 1]
        assert corr > 0.5, (
            f"realized_garch vs arch GARCH(1,1) correlation = {corr:.3f} < 0.5"
        )
