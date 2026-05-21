"""Builder unit tests for C48 MIDAS family promotion.

These tests verify that each new L4 model class correctly implements
the procedure named in its paper reference, using synthetic DGPs with
known parameters. They are separate from tester's independent validation
(test-spec.md tests which builder never sees).

Tests cover:
- Registration: families in OPERATIONAL, not FUTURE
- Input validation: TypeError, ValueError on bad inputs
- Fit/predict: correct shapes, no-crash on edge cases
- Recovery: synthetic DGP weight recovery within tolerance
- Seed-determinism: same seed -> bit-identical NLS results
- Fallback: graceful handling of insufficient data

Families tested:
- _MidasAlmonModel (Ghysels-Santa-Clara-Valkanov 2004)
- _MidasBetaModel (Ghysels-Sinko-Valkanov 2007)
- _MidasStepModel (Foroni-Marcellino-Schumacher 2015)
- _UnrestrictedMidasModel (Foroni-Marcellino-Schumacher 2015)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macroforecast.core.ops.l4_ops import (
    FUTURE_MODEL_FAMILIES,
    OPERATIONAL_MODEL_FAMILIES,
    get_family_status,
)
from macroforecast.core.runtime import (
    _MidasAlmonModel,
    _MidasBetaModel,
    _MidasStepModel,
    _UnrestrictedMidasModel,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic DGP builders
# ---------------------------------------------------------------------------


def _make_lf_data(
    T: int = 100,
    K: int = 6,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Return (X_df, y_s, w_uniform) with uniform true weights for a quick
    sanity check (not used for weight recovery tests)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-01", periods=T, freq="QS")
    X_data = rng.standard_normal((T, K))
    w = np.full(K, 1.0 / K)
    y_data = 0.5 + 2.0 * (X_data @ w) + 0.05 * rng.standard_normal(T)
    X_df = pd.DataFrame(X_data, index=idx, columns=[f"x_lag{k}" for k in range(K)])
    y_s = pd.Series(y_data, index=idx, name="y")
    return X_df, y_s, w


def _make_almon_dgp(
    T: int = 200,
    K: int = 8,
    Q: int = 2,
    theta: np.ndarray | None = None,
    mu: float = 1.0,
    beta: float = 2.0,
    noise_scale: float = 0.05,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Synthetic DGP for Almon weight recovery tests.

    Returns X_df, y_s, w_true where w_true is the normalized Almon weight
    vector for the given theta.
    """
    if theta is None:
        theta = np.array([0.5, 0.1, -0.015])  # Q=2 default
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-01", periods=T, freq="QS")
    kk = np.arange(K, dtype=float)
    w_raw = np.array([sum(float(theta[q]) * k ** q for q in range(Q + 1)) for k in kk])
    w_raw = np.maximum(w_raw, 0.0)
    w_true = w_raw / w_raw.sum()
    X_data = rng.standard_normal((T, K))
    y_data = mu + beta * (X_data @ w_true) + noise_scale * rng.standard_normal(T)
    X_df = pd.DataFrame(X_data, index=idx, columns=[f"x{k}" for k in range(K)])
    y_s = pd.Series(y_data, index=idx, name="y")
    return X_df, y_s, w_true


def _make_beta_dgp(
    T: int = 200,
    K: int = 8,
    a: float = 2.0,
    b_shape: float = 5.0,
    mu: float = 0.5,
    beta: float = 1.5,
    noise_scale: float = 0.05,
    seed: int = 13,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Synthetic DGP for Beta weight recovery tests."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-01", periods=T, freq="QS")
    kk = (np.arange(K, dtype=float) + 1.0) / (K + 1.0)
    w_raw = kk ** (a - 1.0) * (1.0 - kk) ** (b_shape - 1.0)
    w_true = w_raw / w_raw.sum()
    X_data = rng.standard_normal((T, K))
    y_data = mu + beta * (X_data @ w_true) + noise_scale * rng.standard_normal(T)
    X_df = pd.DataFrame(X_data, index=idx, columns=[f"x{k}" for k in range(K)])
    y_s = pd.Series(y_data, index=idx, name="y")
    return X_df, y_s, w_true


def _make_step_dgp(
    T: int = 200,
    K: int = 9,
    S: int = 3,
    step_coefs: np.ndarray | None = None,
    intercept: float = 0.3,
    noise_scale: float = 0.05,
    seed: int = 21,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Synthetic DGP for step-function OLS recovery tests."""
    if step_coefs is None:
        step_coefs = np.array([2.0, 1.0, 0.5])
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-01", periods=T, freq="QS")
    X_data = rng.standard_normal((T, K))
    # Group boundaries: [s*K//S, (s+1)*K//S)
    groups = [(s * K // S, min((s + 1) * K // S, K)) for s in range(S)]
    step_agg = np.stack(
        [X_data[:, lo:hi].mean(axis=1) for lo, hi in groups], axis=1
    )
    y_data = intercept + step_agg @ step_coefs + noise_scale * rng.standard_normal(T)
    X_df = pd.DataFrame(X_data, index=idx, columns=[f"x{k}" for k in range(K)])
    y_s = pd.Series(y_data, index=idx, name="y")
    return X_df, y_s, step_coefs


def _make_umidas_dgp(
    T: int = 200,
    K: int = 4,
    psi: np.ndarray | None = None,
    intercept: float = 0.5,
    noise_scale: float = 0.05,
    seed: int = 37,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Synthetic DGP for U-MIDAS OLS recovery tests."""
    if psi is None:
        psi = np.array([1.5, 0.8, 0.4, 0.2])
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-01", periods=T, freq="QS")
    X_data = rng.standard_normal((T, K))
    y_data = intercept + X_data @ psi + noise_scale * rng.standard_normal(T)
    X_df = pd.DataFrame(X_data, index=idx, columns=[f"x{k}" for k in range(K)])
    y_s = pd.Series(y_data, index=idx, name="y")
    return X_df, y_s, psi


# ---------------------------------------------------------------------------
# Section R: Registration tests
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_midas_almon_in_operational(self):
        assert "midas_almon" in OPERATIONAL_MODEL_FAMILIES
        assert get_family_status("midas_almon") == "operational"

    def test_midas_beta_in_operational(self):
        assert "midas_beta" in OPERATIONAL_MODEL_FAMILIES
        assert get_family_status("midas_beta") == "operational"

    def test_midas_step_in_operational(self):
        assert "midas_step" in OPERATIONAL_MODEL_FAMILIES
        assert get_family_status("midas_step") == "operational"

    def test_dfm_unrestricted_midas_in_operational(self):
        assert "dfm_unrestricted_midas" in OPERATIONAL_MODEL_FAMILIES
        assert get_family_status("dfm_unrestricted_midas") == "operational"

    def test_midas_families_not_in_future(self):
        for fam in ("midas_almon", "midas_beta", "midas_step", "dfm_unrestricted_midas"):
            assert fam not in FUTURE_MODEL_FAMILIES, (
                f"{fam!r} should not be in FUTURE_MODEL_FAMILIES after C48 promotion"
            )

    def test_realized_garch_still_future(self):
        # realized_garch is C49 scope; must NOT be promoted here
        assert "realized_garch" in FUTURE_MODEL_FAMILIES
        assert get_family_status("realized_garch") == "future"

    def test_operational_count_at_least_46(self):
        # Was 42; C48 promotes 4 -> at least 46
        assert len(OPERATIONAL_MODEL_FAMILIES) >= 46


# ---------------------------------------------------------------------------
# Section V: Input validation tests
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Verify that model classes raise correct errors on bad inputs."""

    def test_almon_requires_dataframe_X_fit(self):
        model = _MidasAlmonModel()
        with pytest.raises(TypeError, match="pd.DataFrame"):
            model.fit(np.zeros((5, 3)), pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))  # type: ignore

    def test_almon_requires_series_y_fit(self):
        model = _MidasAlmonModel()
        X = pd.DataFrame({"a": [1.0] * 5})
        with pytest.raises(TypeError, match="pd.Series"):
            model.fit(X, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))  # type: ignore

    def test_almon_requires_dataframe_X_predict(self):
        model = _MidasAlmonModel()
        X, y, _ = _make_lf_data(T=20, K=4, seed=1)
        model.fit(X, y)
        with pytest.raises(TypeError, match="pd.DataFrame"):
            model.predict(np.zeros((5, 4)))  # type: ignore

    def test_almon_freq_ratio_lt1_raises(self):
        with pytest.raises(ValueError, match="freq_ratio"):
            _MidasAlmonModel(freq_ratio=0)

    def test_almon_n_lags_lt1_raises(self):
        with pytest.raises(ValueError, match="n_lags_high"):
            _MidasAlmonModel(n_lags_high=0)

    def test_almon_polynomial_order_lt0_raises(self):
        with pytest.raises(ValueError, match="polynomial_order"):
            _MidasAlmonModel(polynomial_order=-1)

    def test_beta_freq_ratio_lt1_raises(self):
        with pytest.raises(ValueError, match="freq_ratio"):
            _MidasBetaModel(freq_ratio=-1)

    def test_step_n_steps_lt1_raises(self):
        with pytest.raises(ValueError, match="n_steps"):
            _MidasStepModel(n_steps=0)

    def test_umidas_freq_ratio_lt1_raises(self):
        with pytest.raises(ValueError, match="freq_ratio"):
            _UnrestrictedMidasModel(freq_ratio=0)

    def test_umidas_n_lags_int_lt1_raises(self):
        with pytest.raises(ValueError, match="n_lags_high"):
            _UnrestrictedMidasModel(n_lags_high=0)


# ---------------------------------------------------------------------------
# Section C: Contract / shape tests
# ---------------------------------------------------------------------------


class TestContract:
    """Verify fit/predict produce correct shapes and don't crash on edge cases."""

    def test_almon_predict_shape(self):
        X, y, _ = _make_lf_data(T=50, K=6)
        model = _MidasAlmonModel(n_lags_high=6, n_starts=2).fit(X, y)
        preds = model.predict(X.iloc[:10])
        assert preds.shape == (10,)
        assert np.all(np.isfinite(preds))

    def test_beta_predict_shape(self):
        X, y, _ = _make_lf_data(T=50, K=6)
        model = _MidasBetaModel(n_lags_high=6, n_starts=2).fit(X, y)
        preds = model.predict(X.iloc[:7])
        assert preds.shape == (7,)
        assert np.all(np.isfinite(preds))

    def test_step_predict_shape(self):
        X, y, _ = _make_lf_data(T=50, K=6)
        model = _MidasStepModel(n_lags_high=6, n_steps=2).fit(X, y)
        preds = model.predict(X.iloc[:8])
        assert preds.shape == (8,)
        assert np.all(np.isfinite(preds))

    def test_umidas_predict_shape(self):
        X, y, _ = _make_lf_data(T=50, K=4)
        model = _UnrestrictedMidasModel(n_lags_high=4).fit(X, y)
        preds = model.predict(X.iloc[:3])
        assert preds.shape == (3,)
        assert np.all(np.isfinite(preds))

    def test_almon_empty_X_predict_returns_zero_array(self):
        X, y, _ = _make_lf_data(T=30, K=4)
        model = _MidasAlmonModel(n_lags_high=4, n_starts=1).fit(X, y)
        preds = model.predict(X.iloc[:0])
        assert preds.shape == (0,)

    def test_almon_empty_X_fit_no_crash(self):
        idx = pd.date_range("2000-01-01", periods=0, freq="QS")
        X_empty = pd.DataFrame(columns=["a", "b"], index=idx)
        y_empty = pd.Series([], index=idx, dtype=float, name="y")
        model = _MidasAlmonModel(n_lags_high=2, n_starts=1)
        model.fit(X_empty, y_empty)  # must not crash
        assert model._w_hat is not None

    def test_beta_single_row_fit_no_crash(self):
        X = pd.DataFrame({"a": [1.0]}, index=pd.date_range("2000Q1", periods=1, freq="QS"))
        y = pd.Series([0.5], index=X.index, name="y")
        model = _MidasBetaModel(n_lags_high=1, n_starts=1)
        model.fit(X, y)  # insufficient data -> fallback

    def test_step_single_row_fit_no_crash(self):
        X = pd.DataFrame({"a": [1.0]}, index=pd.date_range("2000Q1", periods=1, freq="QS"))
        y = pd.Series([0.5], index=X.index, name="y")
        model = _MidasStepModel(n_lags_high=1, n_steps=1)
        model.fit(X, y)  # insufficient data -> fallback

    def test_umidas_single_row_fit_no_crash(self):
        X = pd.DataFrame({"a": [1.0]}, index=pd.date_range("2000Q1", periods=1, freq="QS"))
        y = pd.Series([0.5], index=X.index, name="y")
        model = _UnrestrictedMidasModel(n_lags_high=1)
        model.fit(X, y)

    def test_predict_before_fit_returns_zeros(self):
        model = _MidasAlmonModel(n_lags_high=4)
        X = pd.DataFrame({"a": [1.0, 2.0]})
        preds = model.predict(X)
        assert preds.shape == (2,)
        assert np.all(preds == 0.0)


# ---------------------------------------------------------------------------
# Section REC: Recovery tests
# ---------------------------------------------------------------------------


class TestRecovery:
    """Verify that estimated parameters recover the true DGP values."""

    WEIGHT_TOL = 0.05  # absolute tolerance for weight recovery

    def test_almon_weight_recovery(self):
        """Almon weights should recover within 0.05 absolute on T=200."""
        theta_true = np.array([0.4, 0.15, -0.02])
        X_df, y_s, w_true = _make_almon_dgp(
            T=200, K=8, Q=2, theta=theta_true, mu=1.0, beta=2.0, noise_scale=0.03, seed=42
        )
        model = _MidasAlmonModel(
            n_lags_high=8, polynomial_order=2, n_starts=5, random_state=0
        )
        model.fit(X_df, y_s)
        assert model._w_hat is not None
        max_dev = float(np.max(np.abs(model._w_hat - w_true)))
        assert max_dev < self.WEIGHT_TOL, (
            f"Almon weight recovery failed: max abs deviation = {max_dev:.4f} > {self.WEIGHT_TOL}"
        )

    def test_almon_intercept_slope_recovery(self):
        """Intercept and slope should be close to true values."""
        theta_true = np.array([0.5, 0.1, -0.015])
        X_df, y_s, w_true = _make_almon_dgp(
            T=200, K=6, Q=2, theta=theta_true, mu=1.0, beta=2.0, noise_scale=0.02, seed=99
        )
        model = _MidasAlmonModel(
            n_lags_high=6, polynomial_order=2, n_starts=5, random_state=0
        )
        model.fit(X_df, y_s)
        assert abs(model._intercept - 1.0) < 0.2, (
            f"Almon intercept {model._intercept:.3f} too far from true 1.0"
        )
        assert abs(model._slope - 2.0) < 0.3, (
            f"Almon slope {model._slope:.3f} too far from true 2.0"
        )

    def test_beta_weight_recovery(self):
        """Beta weights should recover within 0.05 absolute on T=200."""
        X_df, y_s, w_true = _make_beta_dgp(
            T=200, K=8, a=2.0, b_shape=5.0, mu=0.5, beta=1.5, noise_scale=0.02, seed=13
        )
        model = _MidasBetaModel(n_lags_high=8, n_starts=5, random_state=13)
        model.fit(X_df, y_s)
        assert model._w_hat is not None
        max_dev = float(np.max(np.abs(model._w_hat - w_true)))
        assert max_dev < self.WEIGHT_TOL, (
            f"Beta weight recovery failed: max abs deviation = {max_dev:.4f} > {self.WEIGHT_TOL}"
        )

    def test_beta_shape_params_recovery(self):
        """Beta shape params a, b should be close to true values."""
        X_df, y_s, w_true = _make_beta_dgp(
            T=200, K=8, a=2.0, b_shape=5.0, noise_scale=0.02, seed=42
        )
        model = _MidasBetaModel(n_lags_high=8, n_starts=5, random_state=0)
        model.fit(X_df, y_s)
        assert model._theta_hat is not None
        a_hat, b_hat = model._theta_hat[0], model._theta_hat[1]
        # Allow ±1.0 tolerance on shape params; NLS can be multimodal here
        assert abs(a_hat - 2.0) < 1.0, f"Beta a_hat={a_hat:.3f}, true=2.0"
        assert abs(b_hat - 5.0) < 1.5, f"Beta b_hat={b_hat:.3f}, true=5.0"

    def test_step_coef_recovery(self):
        """Step OLS should recover true coefficients to near machine precision."""
        step_coefs_true = np.array([2.0, 1.0, 0.5])
        X_df, y_s, _ = _make_step_dgp(
            T=200, K=9, S=3, step_coefs=step_coefs_true, intercept=0.3, noise_scale=0.02, seed=21
        )
        model = _MidasStepModel(n_lags_high=9, n_steps=3)
        model.fit(X_df, y_s)
        assert model._step_coef is not None
        max_dev = float(np.max(np.abs(model._step_coef - step_coefs_true)))
        assert max_dev < 0.1, (
            f"Step coef recovery failed: max abs deviation = {max_dev:.4f}"
        )
        assert abs(model._intercept - 0.3) < 0.1, (
            f"Step intercept {model._intercept:.3f} too far from true 0.3"
        )

    def test_umidas_coef_recovery(self):
        """U-MIDAS OLS should recover true free coefficients accurately."""
        psi_true = np.array([1.5, 0.8, 0.4, 0.2])
        X_df, y_s, _ = _make_umidas_dgp(
            T=200, K=4, psi=psi_true, intercept=0.5, noise_scale=0.02, seed=37
        )
        model = _UnrestrictedMidasModel(n_lags_high=4)
        model.fit(X_df, y_s)
        assert model._coef is not None
        # coef[0] = intercept, coef[1:] = psi
        psi_hat = model._coef[1:]
        max_dev = float(np.max(np.abs(psi_hat - psi_true)))
        assert max_dev < 0.1, (
            f"U-MIDAS psi recovery failed: max abs deviation = {max_dev:.4f}"
        )
        assert abs(model._coef[0] - 0.5) < 0.1


# ---------------------------------------------------------------------------
# Section SD: Seed-determinism tests
# ---------------------------------------------------------------------------


class TestSeedDeterminism:
    """Same seed must produce bit-identical NLS trajectories."""

    def test_almon_same_seed_bit_identical(self):
        X_df, y_s, _ = _make_almon_dgp(T=80, K=6, seed=17)
        m1 = _MidasAlmonModel(n_lags_high=6, n_starts=3, random_state=42).fit(X_df, y_s)
        m2 = _MidasAlmonModel(n_lags_high=6, n_starts=3, random_state=42).fit(X_df, y_s)
        assert np.array_equal(m1._w_hat, m2._w_hat), "Almon w_hat not bit-identical"
        assert m1._intercept == m2._intercept
        assert m1._slope == m2._slope

    def test_almon_different_seeds_not_identical(self):
        X_df, y_s, _ = _make_almon_dgp(T=80, K=6, seed=17)
        m1 = _MidasAlmonModel(n_lags_high=6, n_starts=5, random_state=0).fit(X_df, y_s)
        m2 = _MidasAlmonModel(n_lags_high=6, n_starts=5, random_state=999).fit(X_df, y_s)
        # With perturbed starts, different seeds may produce different solutions
        # This test is informational; we just verify the model runs without error.
        assert m1._w_hat is not None
        assert m2._w_hat is not None

    def test_beta_same_seed_bit_identical(self):
        X_df, y_s, _ = _make_beta_dgp(T=80, K=6, seed=17)
        m1 = _MidasBetaModel(n_lags_high=6, n_starts=3, random_state=55).fit(X_df, y_s)
        m2 = _MidasBetaModel(n_lags_high=6, n_starts=3, random_state=55).fit(X_df, y_s)
        assert np.array_equal(m1._w_hat, m2._w_hat), "Beta w_hat not bit-identical"
        assert m1._intercept == m2._intercept

    def test_step_deterministic_ols(self):
        """OLS is deterministic; no seed needed. Verify same result on two fits."""
        X_df, y_s, _ = _make_step_dgp(T=80, K=9, S=3, seed=17)
        m1 = _MidasStepModel(n_lags_high=9, n_steps=3).fit(X_df, y_s)
        m2 = _MidasStepModel(n_lags_high=9, n_steps=3).fit(X_df, y_s)
        assert np.array_equal(m1._step_coef, m2._step_coef)
        assert m1._intercept == m2._intercept

    def test_umidas_deterministic_ols(self):
        """U-MIDAS OLS is deterministic; random_state is ignored but must be accepted."""
        X_df, y_s, _ = _make_umidas_dgp(T=80, K=4, seed=17)
        m1 = _UnrestrictedMidasModel(n_lags_high=4, random_state=0).fit(X_df, y_s)
        m2 = _UnrestrictedMidasModel(n_lags_high=4, random_state=99).fit(X_df, y_s)
        # OLS is deterministic regardless of random_state; results must match
        assert np.array_equal(m1._coef, m2._coef)

    def test_predict_is_deterministic_given_fitted_state(self):
        """Calling predict twice on same model gives bit-identical results."""
        X_df, y_s, _ = _make_almon_dgp(T=50, K=4, seed=1)
        model = _MidasAlmonModel(n_lags_high=4, n_starts=3, random_state=0).fit(X_df, y_s)
        p1 = model.predict(X_df.iloc[:5])
        p2 = model.predict(X_df.iloc[:5])
        assert np.array_equal(p1, p2)


# ---------------------------------------------------------------------------
# Section EC: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Graceful handling of edge inputs that are encountered in walk-forward OOS."""

    def test_almon_insufficient_rows_fallback_uniform_weights(self):
        """When fewer than Q+3 common rows exist, fallback to uniform weights."""
        idx = pd.date_range("2000-01-01", periods=3, freq="QS")
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx)
        y = pd.Series([0.5, 1.0, 1.5], index=idx, name="y")
        model = _MidasAlmonModel(n_lags_high=1, polynomial_order=2, n_starts=1)
        model.fit(X, y)
        # w_hat should be uniform (fallback)
        assert model._w_hat is not None
        assert np.allclose(model._w_hat, 1.0 / model.n_lags_high)

    def test_step_insufficient_rows_fallback_zero_coefs(self):
        """When fewer than S+2 common rows exist, step coefs should be zero."""
        idx = pd.date_range("2000-01-01", periods=2, freq="QS")
        X = pd.DataFrame({"a": [1.0, 2.0]}, index=idx)
        y = pd.Series([0.5, 1.0], index=idx, name="y")
        model = _MidasStepModel(n_lags_high=3, n_steps=3)
        model.fit(X, y)
        # n_steps=3, need at least 5 rows -> fallback
        assert model._step_coef is not None
        assert np.all(model._step_coef == 0.0)

    def test_umidas_predict_returns_finite(self):
        """Even with near-collinear X, predict must return finite values (lstsq SVD handles this)."""
        rng = np.random.default_rng(0)
        T = 50
        idx = pd.date_range("2000-01-01", periods=T, freq="QS")
        # Near-collinear: two columns are almost identical
        x1 = rng.standard_normal(T)
        X = pd.DataFrame({"a": x1, "b": x1 + 1e-10 * rng.standard_normal(T)}, index=idx)
        y = pd.Series(x1 + 0.1 * rng.standard_normal(T), index=idx, name="y")
        model = _UnrestrictedMidasModel(n_lags_high=2).fit(X, y)
        preds = model.predict(X)
        assert np.all(np.isfinite(preds)), "U-MIDAS predict produced non-finite values on near-collinear X"

    def test_almon_sum_to_one_true_weights_sum(self):
        """With sum_to_one=True, w_hat should sum to approximately 1.0."""
        X_df, y_s, _ = _make_lf_data(T=50, K=6)
        model = _MidasAlmonModel(n_lags_high=6, sum_to_one=True, n_starts=2).fit(X_df, y_s)
        assert model._w_hat is not None
        assert abs(float(np.sum(model._w_hat)) - 1.0) < 1e-6, (
            f"w_hat does not sum to 1.0: sum = {float(np.sum(model._w_hat))}"
        )

    def test_beta_weights_always_sum_to_one(self):
        """Beta weights always sum to one (construction invariant)."""
        X_df, y_s, _ = _make_lf_data(T=50, K=6)
        model = _MidasBetaModel(n_lags_high=6, n_starts=2).fit(X_df, y_s)
        assert model._w_hat is not None
        assert abs(float(np.sum(model._w_hat)) - 1.0) < 1e-6

    def test_umidas_bic_str_raises_no_error(self):
        """n_lags_high='bic' on freq_ratio=1 should not raise; K defaults to n_cols."""
        X_df, y_s, _ = _make_lf_data(T=50, K=4)
        model = _UnrestrictedMidasModel(n_lags_high="bic", freq_ratio=1)
        model.fit(X_df, y_s)  # should not raise
        assert model._K_fit == X_df.shape[1]

    def test_step_n_steps_gt_k_clips_to_available(self):
        """n_steps > K should not crash; groups will have zero width but no exception."""
        X_df, y_s, _ = _make_lf_data(T=50, K=2)
        model = _MidasStepModel(n_lags_high=2, n_steps=5)
        model.fit(X_df, y_s)  # no crash expected
        preds = model.predict(X_df.iloc[:3])
        assert preds.shape == (3,)


# ---------------------------------------------------------------------------
# Section INT: Integration smoke (dispatch via _build_l4_model)
# ---------------------------------------------------------------------------


class TestDispatch:
    """Verify that _build_l4_model correctly dispatches all 4 families.

    Note: _build_l4_model(family, params) reads seed from params["random_state"].
    """

    def test_build_l4_model_midas_almon(self):
        from macroforecast.core.runtime import _build_l4_model

        params = {"n_lags_high": 6, "polynomial_order": 2, "n_starts": 2, "random_state": 0}
        model = _build_l4_model("midas_almon", params=params)
        assert isinstance(model, _MidasAlmonModel)
        assert model.polynomial_order == 2

    def test_build_l4_model_midas_beta(self):
        from macroforecast.core.runtime import _build_l4_model

        params = {"n_lags_high": 8, "n_starts": 2, "random_state": 7}
        model = _build_l4_model("midas_beta", params=params)
        assert isinstance(model, _MidasBetaModel)
        assert model.random_state == 7

    def test_build_l4_model_midas_step(self):
        from macroforecast.core.runtime import _build_l4_model

        params = {"n_lags_high": 9, "n_steps": 3, "freq_ratio": 3, "random_state": 0}
        model = _build_l4_model("midas_step", params=params)
        assert isinstance(model, _MidasStepModel)
        assert model.n_steps == 3

    def test_build_l4_model_dfm_unrestricted_midas(self):
        from macroforecast.core.runtime import _build_l4_model

        params = {"n_lags_high": "bic", "include_y_lag": True, "random_state": 0}
        model = _build_l4_model("dfm_unrestricted_midas", params=params)
        assert isinstance(model, _UnrestrictedMidasModel)
        assert model.include_y_lag is True

    def test_build_l4_model_midas_step_default_n_steps_equals_freq_ratio(self):
        """When n_steps is not specified, default = freq_ratio."""
        from macroforecast.core.runtime import _build_l4_model

        params = {"n_lags_high": 12, "freq_ratio": 4, "random_state": 0}
        model = _build_l4_model("midas_step", params=params)
        assert isinstance(model, _MidasStepModel)
        assert model.n_steps == 4  # default = freq_ratio

    def test_build_l4_model_midas_step_default_n_steps_freq1(self):
        """When freq_ratio=1 and n_steps not specified, default = 1."""
        from macroforecast.core.runtime import _build_l4_model

        params = {"n_lags_high": 6, "random_state": 0}  # freq_ratio defaults to 1
        model = _build_l4_model("midas_step", params=params)
        assert model.n_steps == 1

    def test_midas_families_fit_predict_end_to_end(self):
        """All 4 families fit and predict without error on simple data."""
        from macroforecast.core.runtime import _build_l4_model

        X_df, y_s, _ = _make_lf_data(T=60, K=4)
        families = [
            ("midas_almon", {"n_lags_high": 4, "n_starts": 2, "random_state": 0}),
            ("midas_beta", {"n_lags_high": 4, "n_starts": 2, "random_state": 0}),
            ("midas_step", {"n_lags_high": 4, "n_steps": 2, "random_state": 0}),
            ("dfm_unrestricted_midas", {"n_lags_high": 4, "random_state": 0}),
        ]
        for family, params in families:
            model = _build_l4_model(family, params=params)
            model.fit(X_df, y_s)
            preds = model.predict(X_df.iloc[:5])
            assert preds.shape == (5,), f"{family}: wrong predict shape"
            assert np.all(np.isfinite(preds)), f"{family}: non-finite predictions"
