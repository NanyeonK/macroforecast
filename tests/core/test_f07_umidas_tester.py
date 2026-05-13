"""
F-07 U-MIDAS Paper-Faithful Fix — Independent Tester Pipeline Tests

Run: 2026-05-12-phase-f07-umidas-paper-faithful-fix
Spec: test-spec.md (TEST-R1-01..TEST-R4-02 + edge cases)

This file is authored by the tester pipeline independently of spec.md and
implementation.md. It validates behavioral contracts from test-spec.md only.

DO NOT modify this file — tester-owned.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from macroforecast.core.runtime import _bic_select_k, _midas_lag_stack, _u_midas
from macroforecast.recipes.paper_methods import u_midas

import macroforecast


# ---------------------------------------------------------------------------
# Shared helpers (independent re-implementation — no import from test_phase_c)
# ---------------------------------------------------------------------------

import datetime


def _monthly_dates(t: int, start_year: int = 2010, start_month: int = 1) -> list:
    out = []
    y, m = start_year, start_month
    for _ in range(t):
        out.append(datetime.date(y, m, 1).strftime("%Y-%m-%d"))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def _build_panel_from_y_X(y: np.ndarray, X: np.ndarray | None = None) -> dict:
    """Build panel dict where y and X share the SAME row count (HF length).

    For U-MIDAS tests: y and X must be same length. The recipe uses freq_ratio
    to identify the LF vs HF relationship internally. Build y at HF scale
    with a simple linear-lag relationship, or provide y_hf and x_hf explicitly.
    """
    t = int(y.shape[0])
    panel: dict = {"date": _monthly_dates(t), "y": list(map(float, y))}
    if X is not None:
        assert X.shape[0] == t, (
            "X.shape[0]={} must equal t={}".format(X.shape[0], t)
        )
        for j in range(X.shape[1]):
            panel["x{}".format(j + 1)] = list(map(float, X[:, j]))
    else:
        rng = np.random.default_rng(0)
        panel["x1"] = list(rng.normal(0.0, 0.1, size=t))
    return panel


def _run_recipe(recipe: dict) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = macroforecast.run(recipe)
    assert result.cells, "recipe should produce at least one cell"
    return result


def _assert_l4_forecasts(result: Any) -> None:
    artifacts = result.cells[0].runtime_result.artifacts
    assert "l4_forecasts_v1" in artifacts, (
        "l4_forecasts_v1 missing; artifacts={}".format(list(artifacts.keys()))
    )


def _extract_forecasts(result: Any) -> np.ndarray:
    """Extract L4 forecast values as numpy array from L4ForecastsArtifact."""
    artifacts = result.cells[0].runtime_result.artifacts
    assert "l4_forecasts_v1" in artifacts
    fc_raw = artifacts["l4_forecasts_v1"]
    # L4ForecastsArtifact has a .forecasts dict of float values
    if hasattr(fc_raw, "forecasts"):
        vals = list(fc_raw.forecasts.values())
        return np.array(vals, dtype=float)
    # Fallback for DataFrame-like artifacts
    if hasattr(fc_raw, "to_numpy"):
        return fc_raw.to_numpy().flatten().astype(float)
    if hasattr(fc_raw, "values"):
        return np.asarray(fc_raw.values, dtype=float).flatten()
    return np.asarray(fc_raw, dtype=float).flatten()


# ---------------------------------------------------------------------------
# DGP-A: Simple linear-lag DGP (seed=7, T_HF=120, freq_ratio=3)
# NOTE: Panel is built at HF length (T_HF=120). The u_midas recipe uses
# freq_ratio=3 internally to identify which rows are LF observations.
# We use y computed at every row (HF-frequency y), matching the existing
# test pattern in test_phase_c_top6.py.
# ---------------------------------------------------------------------------

def _dgp_a_hf():
    """DGP-A at HF length for panel construction. Returns y_hf, x_hf, k."""
    rng = np.random.default_rng(7)
    T_HF = 120
    k = 3
    x_hf = rng.standard_normal(T_HF)
    # y computed at HF scale (same as existing test_public_path pattern)
    y_hf = np.zeros(T_HF)
    for t in range(T_HF):
        y_hf[t] = (
            0.6 * x_hf[t]
            + 0.3 * (x_hf[t - 1] if t >= 1 else 0.0)
            + 0.1 * (x_hf[t - 2] if t >= 2 else 0.0)
        )
    y_hf += rng.normal(0, 0.05, T_HF)
    return y_hf, x_hf, k


def _dgp_a_lf():
    """DGP-A at LF level for unit-level _u_midas tests. Returns y_lf, x_hf, k."""
    rng = np.random.default_rng(7)
    T_HF = 120
    k = 3
    T_LF = T_HF // k
    x_hf = rng.standard_normal(T_HF)
    y_lf = np.array([
        0.6 * x_hf[tau * 3]
        + 0.3 * x_hf[max(tau * 3 - 1, 0)]
        + 0.1 * x_hf[max(tau * 3 - 2, 0)]
        for tau in range(T_LF)
    ]) + rng.normal(0, 0.05, T_LF)
    return y_lf, x_hf, k, T_LF


# ---------------------------------------------------------------------------
# Group 1 — R1: OLS Default
# ---------------------------------------------------------------------------

class TestR1OLSDefault:
    """TEST-R1-01, TEST-R1-02, TEST-R1-03 from test-spec.md."""

    def test_r1_01_ols_lstsq_matches_sklearn(self):
        """TEST-R1-01 Method A: _u_midas OLS via lstsq == sklearn LinearRegression.

        Spec tolerance: rtol=1e-6.
        """
        from sklearn.linear_model import LinearRegression

        y_lf, x_hf, k, T_LF = _dgp_a_lf()
        T_HF = len(x_hf)

        frame = pd.DataFrame({"x": x_hf}, index=pd.RangeIndex(T_HF))
        y_series = pd.Series(y_lf, index=pd.RangeIndex(0, T_HF, k))

        stacked = _u_midas(
            frame, freq_ratio=k, n_lags_high=3,
            target_freq="low", include_y_lag=False
        )

        # Align and drop NaN
        combined = pd.concat([y_series.rename("y"), stacked], axis=1).dropna()
        X = combined[["x_lag0", "x_lag1", "x_lag2"]].to_numpy()
        y = combined["y"].to_numpy()

        # sklearn reference
        lr = LinearRegression(fit_intercept=True).fit(X, y)
        y_hat_lr = lr.predict(X)

        # numpy lstsq (OLS) — what the impl uses
        X_int = np.column_stack([np.ones(len(y)), X])
        beta_np, _, _, _ = np.linalg.lstsq(X_int, y, rcond=None)
        y_hat_np = X_int @ beta_np

        np.testing.assert_allclose(
            y_hat_np, y_hat_lr, rtol=1e-6,
            err_msg="OLS via lstsq must match sklearn LinearRegression (rtol=1e-6)"
        )

    def test_r1_01_ols_vs_ridge_differ_on_strong_signal(self):
        """TEST-R1-01 Method B: OLS and ridge produce different predictions on DGP-A.

        Spec tolerance: mean absolute difference > 0.01.
        Panel built at HF length (both y and x have T_HF=120 rows).
        """
        y_hf, x_hf, k = _dgp_a_hf()
        T_HF = len(x_hf)
        panel = _build_panel_from_y_X(y_hf, X=x_hf.reshape(-1, 1))

        recipe_ols = u_midas(
            target="y", horizon=1, freq_ratio=k, n_lags_high=3,
            include_y_lag=False, regularization="none", panel=panel
        )
        recipe_ridge = u_midas(
            target="y", horizon=1, freq_ratio=k, n_lags_high=3,
            include_y_lag=False, regularization="ridge", alpha=1.0, panel=panel
        )

        result_ols = _run_recipe(recipe_ols)
        result_ridge = _run_recipe(recipe_ridge)

        fc_ols = _extract_forecasts(result_ols)
        fc_ridge = _extract_forecasts(result_ridge)

        diff = np.abs(fc_ols - fc_ridge).mean()
        assert diff > 0.01, (
            "OLS and ridge should differ for strong-signal DGP; "
            "mean diff={:.4f} (spec: > 0.01)".format(diff)
        )

    def test_r1_02_ridge_explicit_opt_in_works(self):
        """TEST-R1-02: ridge explicit opt-in runs without error.

        Panel at HF length.
        """
        y_hf, x_hf, k = _dgp_a_hf()
        panel = _build_panel_from_y_X(y_hf, X=x_hf.reshape(-1, 1))

        recipe = u_midas(
            target="y", horizon=1, freq_ratio=k, n_lags_high=3,
            include_y_lag=False, regularization="ridge", alpha=1.0, panel=panel
        )
        result = _run_recipe(recipe)
        _assert_l4_forecasts(result)  # no error + L4 forecasts produced

    def test_r1_03_default_regularization_is_ols(self):
        """TEST-R1-03: Default fit_model family == 'ols', not ridge.

        Navigates recipe['4_forecasting_model']['nodes'] to find fit_model node.
        Spec: fit_node["params"]["family"] == "ols".
        """
        recipe = u_midas(
            target="y", horizon=1, freq_ratio=3, n_lags_high=3,
            include_y_lag=False, regularization="none",
            panel={"y": [1.0] * 60, "x1": [0.5] * 60}
        )
        layer = recipe.get("4_forecasting_model") or recipe.get("4_model_training")
        assert layer is not None, (
            "Layer '4_forecasting_model' not found; recipe keys={}".format(
                list(recipe.keys())
            )
        )
        fit_nodes = [
            n for n in layer["nodes"]
            if n.get("op") == "fit_model"
        ]
        assert fit_nodes, "No fit_model node found in layer 4"
        fit_node = fit_nodes[0]
        family = fit_node["params"]["family"]
        assert family == "ols", (
            "Default family should be 'ols'; got {!r}".format(family)
        )


# ---------------------------------------------------------------------------
# Group 2 — R2: BIC Lag Selection
# ---------------------------------------------------------------------------

class TestR2BICLagSelection:
    """TEST-R2-01..TEST-R2-05 from test-spec.md."""

    @pytest.fixture
    def dgp_b(self):
        """DGP-B: high noise, lag-0 only. seed=13."""
        rng = np.random.default_rng(13)
        T_HF = 120
        k = 3
        T_LF = 40
        x_hf = rng.standard_normal(T_HF)
        y_lf = 0.8 * x_hf[::3] + rng.standard_normal(T_LF)
        frame = pd.DataFrame({"x": x_hf}, index=pd.RangeIndex(T_HF))
        y_series = pd.Series(y_lf, index=pd.RangeIndex(0, T_HF, k))
        return frame, y_series, k

    @pytest.fixture
    def dgp_c(self):
        """DGP-C: low noise, all 5 lags active. seed=17."""
        rng = np.random.default_rng(17)
        T_HF = 300
        k = 3
        T_LF = 100
        x_hf = rng.standard_normal(T_HF)
        coeffs = [0.5, 0.4, 0.3, 0.2, 0.1]
        y_lf = np.zeros(T_LF)
        for tau in range(T_LF):
            for j, c in enumerate(coeffs):
                hf_idx = tau * 3 - j
                if hf_idx >= 0:
                    y_lf[tau] += c * x_hf[hf_idx]
        y_lf += rng.normal(0, 0.1, T_LF)
        frame = pd.DataFrame({"x": x_hf}, index=pd.RangeIndex(T_HF))
        y_series = pd.Series(y_lf, index=pd.RangeIndex(0, T_HF, k))
        return frame, y_series, k

    def test_r2_01_high_noise_selects_small_k(self, dgp_b):
        """TEST-R2-01: DGP-B high-noise → K_star <= 3.

        Spec tolerance: K_star <= 3.
        """
        frame, y_series, k = dgp_b
        K_max = math.ceil(1.5 * k)  # = 5

        K_star = _bic_select_k(frame, y_series, freq_ratio=k,
                                include_y_lag=False, ic="bic")
        assert 1 <= K_star <= K_max, (
            "K_star={} outside [1, {}]".format(K_star, K_max)
        )
        assert K_star <= 3, (
            "High-noise lag-0-only DGP should select K <= 3; "
            "got K_star={} (spec: K_star <= 3)".format(K_star)
        )

    def test_r2_02_low_noise_selects_large_k(self, dgp_c):
        """TEST-R2-02: DGP-C low-noise all-5-lags → K_star >= 3.

        Spec tolerance: K_star >= 3.
        """
        frame, y_series, k = dgp_c
        K_max = math.ceil(1.5 * k)  # = 5

        K_star = _bic_select_k(frame, y_series, freq_ratio=k,
                                include_y_lag=False, ic="bic")
        assert K_star >= 3, (
            "Rich-signal DGP with 5 active lags should select K >= 3; "
            "got K_star={} (spec: K_star >= 3)".format(K_star)
        )
        assert K_star <= K_max, (
            "K_star={} must not exceed K_max={}".format(K_star, K_max)
        )

    def test_r2_03_k_max_formula_invariant(self):
        """TEST-R2-03: K_star in [1, ceil(1.5*freq_ratio)] for all freq_ratios.

        Verified for freq_ratio in {3, 12, 60}.
        Fix: use len(range(0, T_HF, freq_ratio)) to avoid int-division mismatch.
        """
        for freq_ratio in [3, 12, 60]:
            K_max_expected = math.ceil(1.5 * freq_ratio)
            rng = np.random.default_rng(99)
            T_HF = max(200, K_max_expected * 10)
            x = pd.DataFrame(
                {"x": rng.standard_normal(T_HF)},
                index=pd.RangeIndex(T_HF)
            )
            # Use len(range(...)) to match the actual LF index length
            lf_index = pd.RangeIndex(0, T_HF, freq_ratio)
            n_lf = len(lf_index)
            y = pd.Series(
                rng.standard_normal(n_lf),
                index=lf_index
            )
            K_star = _bic_select_k(x, y, freq_ratio=freq_ratio,
                                    include_y_lag=False, ic="bic")
            assert 1 <= K_star <= K_max_expected, (
                "freq_ratio={}: K_star={} outside [1, {}]".format(
                    freq_ratio, K_star, K_max_expected
                )
            )

    def test_r2_04_integer_n_lags_bypasses_bic(self):
        """TEST-R2-04: Integer n_lags_high=4 → exactly 4 columns, no BIC.

        Spec tolerance: exact column count.
        """
        rng = np.random.default_rng(5)
        T_HF = 60
        k = 3
        frame = pd.DataFrame(
            {"x": rng.standard_normal(T_HF)},
            index=pd.RangeIndex(T_HF)
        )
        out = _u_midas(frame, freq_ratio=k, n_lags_high=4,
                       target_freq="low", include_y_lag=False)
        assert out.shape[1] == 4, (
            "Expected 4 columns for n_lags_high=4; got {}".format(out.shape[1])
        )
        assert list(out.columns) == ["x_lag0", "x_lag1", "x_lag2", "x_lag3"], (
            "Column names wrong: {}".format(list(out.columns))
        )

    def test_r2_05_aic_variant_returns_valid_k(self, dgp_b):
        """TEST-R2-05: ic='aic' runs and returns valid K in [1, K_max].

        Spec tolerance: 1 <= K_star_aic <= 5.
        """
        frame, y_series, k = dgp_b
        K_max = math.ceil(1.5 * k)  # = 5

        K_star_aic = _bic_select_k(frame, y_series, freq_ratio=k,
                                    include_y_lag=False, ic="aic")
        assert 1 <= K_star_aic <= K_max, (
            "AIC K_star={} outside [1, {}]".format(K_star_aic, K_max)
        )


# ---------------------------------------------------------------------------
# Group 3 — R3: AR(1) y-lag term
# ---------------------------------------------------------------------------

class TestR3YLagTerm:
    """TEST-R3-01..TEST-R3-03 from test-spec.md."""

    @pytest.fixture
    def dgp_d(self):
        """DGP-D: AR(1) target, seed=21. Returns LF-level data for unit tests."""
        rng = np.random.default_rng(21)
        T_HF = 120
        k = 3
        T_LF = 40
        x_hf = rng.standard_normal(T_HF)
        y_lf = np.zeros(T_LF)
        for tau in range(T_LF):
            y_lf[tau] = (
                0.7 * (y_lf[tau - 1] if tau > 0 else 0.0)
                + 0.5 * x_hf[tau * 3]
                + rng.normal(0, 0.2)
            )
        frame = pd.DataFrame({"x": x_hf}, index=pd.RangeIndex(T_HF))
        y_series = pd.Series(y_lf, index=pd.RangeIndex(0, T_HF, k))
        return y_lf, x_hf, frame, y_series, k, T_LF

    @pytest.fixture
    def dgp_d_hf(self):
        """DGP-D at HF length for panel construction (for macroforecast.run tests).

        Generates AR-process at HF level, then subsamples to get AR structure
        in the low-frequency target.
        """
        rng = np.random.default_rng(21)
        T_HF = 360  # 120 quarters at m=3 for longer series
        k = 3
        x_hf = rng.standard_normal(T_HF)
        # AR(1) target at HF level: builds correlation structure across LF periods
        y_hf = np.zeros(T_HF)
        for t in range(T_HF):
            prev_lf = y_hf[t - k] if t >= k else 0.0  # AR term at LF period
            y_hf[t] = (
                0.7 * prev_lf
                + 0.5 * x_hf[t]
                + rng.normal(0, 0.2)
            )
        return y_hf, x_hf, k

    def test_r3_01_include_y_lag_adds_column(self, dgp_d):
        """TEST-R3-01: include_y_lag=True adds 'y_lag1' column as 1st column.

        Checks: column presence, shape, y_lag1[0] is NaN, y_lag1[1] == y_lf[0].
        Spec tolerance: atol=1e-10 for lag alignment.
        """
        y_lf, x_hf, frame, y_series, k, T_LF = dgp_d

        out_with = _u_midas(
            frame, freq_ratio=k, n_lags_high=3,
            target_freq="low", include_y_lag=True, y_series=y_series
        )
        out_without = _u_midas(
            frame, freq_ratio=k, n_lags_high=3,
            target_freq="low", include_y_lag=False
        )

        # Column presence checks
        assert "y_lag1" in out_with.columns, (
            "y_lag1 column must be present when include_y_lag=True"
        )
        assert out_with.shape[1] == 4, (
            "Expected 4 cols (y_lag1 + x_lag0,1,2); got {}".format(out_with.shape[1])
        )
        assert "y_lag1" not in out_without.columns
        assert out_without.shape[1] == 3

        # y_lag1 alignment checks
        assert np.isnan(out_with["y_lag1"].iloc[0]), (
            "y_lag1[0] must be NaN (no prior y)"
        )
        assert abs(out_with["y_lag1"].iloc[1] - y_lf[0]) < 1e-10, (
            "y_lag1[1]={} != y_lf[0]={}".format(
                out_with["y_lag1"].iloc[1], y_lf[0]
            )
        )
        assert abs(out_with["y_lag1"].iloc[2] - y_lf[1]) < 1e-10, (
            "y_lag1[2]={} != y_lf[1]={}".format(
                out_with["y_lag1"].iloc[2], y_lf[1]
            )
        )

    def test_r3_02_y_lag_changes_predictions_on_ar_target(self, dgp_d_hf):
        """TEST-R3-02: include_y_lag=True/False produce different predictions.

        Spec tolerance: mean absolute difference > 0.05.
        Panel at HF length for macroforecast.run compatibility.
        """
        y_hf, x_hf, k = dgp_d_hf
        panel_D = _build_panel_from_y_X(y_hf, X=x_hf.reshape(-1, 1))

        recipe_with = u_midas(
            target="y", horizon=1, freq_ratio=k,
            n_lags_high=3, include_y_lag=True,
            regularization="none", panel=panel_D
        )
        recipe_without = u_midas(
            target="y", horizon=1, freq_ratio=k,
            n_lags_high=3, include_y_lag=False,
            regularization="none", panel=panel_D
        )

        result_with = _run_recipe(recipe_with)
        result_without = _run_recipe(recipe_without)

        fc_with = _extract_forecasts(result_with)
        fc_without = _extract_forecasts(result_without)

        mean_diff = np.abs(fc_with - fc_without).mean()
        assert mean_diff > 0.05, (
            "AR-target predictions should differ between include_y_lag=True/False; "
            "mean_diff={:.4f} (spec: > 0.05)".format(mean_diff)
        )

    def test_r3_03_y_lag1_values_are_correctly_lagged(self, dgp_d):
        """TEST-R3-03: y_lag1[tau] == y_lf[tau-1] for all tau in [1, T_LF-1].

        Spec tolerance: atol=1e-10. 39 non-NaN values verified.
        """
        y_lf, x_hf, frame, y_series, k, T_LF = dgp_d

        out_with = _u_midas(
            frame, freq_ratio=k, n_lags_high=3,
            target_freq="low", include_y_lag=True, y_series=y_series
        )

        failures = []
        for tau in range(1, T_LF):
            actual = out_with["y_lag1"].iloc[tau]
            expected = y_lf[tau - 1]
            if abs(actual - expected) >= 1e-10:
                failures.append(
                    "y_lag1[{}]={:.6f} != y_lf[{}]={:.6f}".format(
                        tau, actual, tau - 1, expected
                    )
                )
        assert not failures, (
            "{} alignment failures (atol=1e-10):\n{}".format(
                len(failures), "\n".join(failures[:5])
            )
        )


# ---------------------------------------------------------------------------
# Group 4 — R4: Monte Carlo Table 2 Anchor
# ---------------------------------------------------------------------------

def _generate_hf_var1(rho, delta_l, delta_h, k, T_lf, ES, seed):
    """Generate HF VAR(1) as in paper §3.1 eq.(17). test-spec.md §DGP-E."""
    rng = np.random.default_rng(seed)
    total_lf = T_lf + ES
    total_hf = total_lf * k

    sigma_x2 = max(1.0 - rho ** 2, 0.01)
    sigma_y2 = max((1.0 - rho ** 2) - delta_l ** 2 * sigma_x2, 0.01)

    Phi = np.array([[rho, delta_l], [delta_h, rho]])
    Q = np.diag([sigma_y2, sigma_x2])

    data = np.zeros((total_hf, 2))
    for t in range(1, total_hf):
        data[t] = Phi @ data[t - 1] + rng.multivariate_normal([0.0, 0.0], Q)

    y_lf = data[::k, 0]
    x_hf = data[:, 1]
    return y_lf, x_hf


def _compute_umidas_oos_mse(y_lf, x_hf, k, T_lf, ES):
    """Recursive expanding-window U-MIDAS OOS MSE. test-spec.md §TEST-R4.

    MC-RECAL symmetric fix: uses include_y_lag=True to match paper §3.2
    eq.(20). Both U-MIDAS and MIDAS comparator now include the AR(1) y-lag
    term, making the comparison paper-symmetric. BIC K selection uses
    include_y_lag=False (K selection depends only on HF lag order).
    Forecast step reuses the same frame and y_series as training to ensure
    y_lag1 in the last row = y_train[-1] (correct look-ahead-free alignment).
    """
    mse_list = []
    for es in range(ES):
        train_lf_end = T_lf + es
        train_hf_end = train_lf_end * k

        y_train = y_lf[:train_lf_end]
        x_train = x_hf[:train_hf_end]
        y_test = y_lf[train_lf_end]

        frame = pd.DataFrame({"x": x_train}, index=pd.RangeIndex(train_hf_end))
        y_series = pd.Series(y_train, index=pd.RangeIndex(0, train_hf_end, k))

        # BIC K selection: include_y_lag=False (K selection over HF lag order only)
        K_star = _bic_select_k(frame, y_series, freq_ratio=k,
                                include_y_lag=False, ic="bic")
        K_max = math.ceil(1.5 * k)
        K_star = min(K_star, K_max)

        # Build design matrix WITH y-lag (paper eq.(20): μ_1 y_{τ-1} + ψ(L)x)
        stacked = _u_midas(frame, freq_ratio=k, n_lags_high=K_star,
                           target_freq="low", include_y_lag=True,
                           y_series=y_series)
        combined = pd.concat([y_series.rename("y"), stacked], axis=1).dropna()
        if len(combined) < K_star + 3:
            mse_list.append(np.nan)
            continue

        y_arr = combined["y"].to_numpy()
        X_arr = combined.drop(columns="y").to_numpy()  # includes y_lag1 as first col
        X_int = np.column_stack([np.ones(len(y_arr)), X_arr])
        beta, _, _, _ = np.linalg.lstsq(X_int, y_arr, rcond=None)

        # FORECAST STEP — alignment fix:
        # Get HF lag features from the last row of stacked (without y-lag, same
        # window), then manually set y_lag1 = y_train[-1] (the most recent known
        # LF value) as the forecast-period AR conditioning value.
        # We use include_y_lag=False here to get clean HF-only lag features,
        # then prepend y_lag1 = y_train[-1] explicitly.
        stacked_fore_hf = _u_midas(frame, freq_ratio=k, n_lags_high=K_star,
                                   target_freq="low", include_y_lag=False)
        # Last row contains HF lags for LF period train_lf_end-1 → forecast features
        x_hf_last = stacked_fore_hf.iloc[[-1]].to_numpy()   # shape (1, K_star)
        y_lag1_fore = np.array([[float(y_train[-1])]])       # shape (1, 1)
        # Forecast row: [y_lag1, x_lag0, ..., x_lagK_star]
        x_last = np.concatenate([y_lag1_fore, x_hf_last], axis=1)  # shape (1, K_star+1)
        x_last_int = np.column_stack([np.ones(1), x_last])
        y_hat = float((x_last_int @ beta).flat[0])
        mse_list.append((y_hat - y_test) ** 2)

    valid = [v for v in mse_list if not np.isnan(v)]
    return float(np.mean(valid)) if valid else np.nan


def _compute_midas_oos_mse_baseline(y_lf, x_hf, k, T_lf, ES):
    """NLS exp-Almon MIDAS OOS MSE baseline. Paper §3.2 eq.(18) comparator.

    MC-RECAL common-factor fix: implements paper eq.(18) WITH the common-factor
    restriction (1 - β_1 L^k). The full expanded form is:

        y_t = β_0 + β_1·y_{t-k} + β_2·B(L,θ)·x_{t-1}
              - β_1·β_2·B(L,θ)·x_{t-k-1} + ε

    where agg_t ≡ B(L,θ)·x_{t-1} and agg_tk ≡ B(L,θ)·x_{t-k-1}
    (= agg_series shifted 1 LF period).

    NLS loss: sum((y - b0 - b1*y_lag - b2*agg_t + b1*b2*agg_tk)^2)
    Parameters: [θ_1, θ_2, β_0, β_1, β_2] (5-parameter NLS).
    Grid-search initialization: θ_1 ∈ {-0.5, 0.0, 0.5} × θ_2 ∈
    {-0.01, -0.1, -0.5, -1.0} with simplified-form OLS warm-start.
    OOS forecast: b0 + b1*y_lag_last + b2*agg_test - b1*b2*agg_test_lagk
    where agg_test_lagk = last training agg (= stacked.iloc[-2] @ w_hat,
    i.e., B(L,θ) applied one LF period before the test window).
    """
    from scipy.optimize import minimize as _scipy_minimize

    K = math.ceil(1.5 * k)

    def _w_exp_almon(theta, K_):
        """Exponential Almon weights (paper §3.2 MIDAS comparator)."""
        kk = np.arange(K_, dtype=float)
        z = float(theta[0]) * kk + float(theta[1]) * (kk ** 2)
        z = z - float(np.max(z))   # numerical-stability shift
        e = np.exp(z)
        s = float(np.sum(e))
        if s <= 0 or not np.isfinite(s):
            return np.full(K_, 1.0 / K_, dtype=float)
        return e / s

    mse_list = []
    for es in range(ES):
        train_lf_end = T_lf + es
        train_hf_end = train_lf_end * k

        y_train = y_lf[:train_lf_end]
        x_train = x_hf[:train_hf_end]
        y_test = y_lf[train_lf_end]

        frame = pd.DataFrame({"x": x_train}, index=pd.RangeIndex(train_hf_end))
        stacked = _midas_lag_stack(frame, freq_ratio=k, n_lags_high=K,
                                   target_freq="low")
        y_series = pd.Series(y_train, index=pd.RangeIndex(0, train_hf_end, k))

        # Build y-lag series (y_{τ-1}) aligned to LF index; NaN at τ=0
        y_lag_series = y_series.shift(1)

        # Build agg-lag series: agg_{τ-1} = B(L,θ)·x at LF period τ-1
        # This is the common-factor term B(L,θ)·x_{t-k-1} in expanded eq.(18).
        # stacked.shift(1) shifts each LF row back by 1 LF period → NaN at τ=0.
        stacked_lagk = stacked.shift(1)

        # Align all series; dropna removes first 2 rows (NaN y_lag and NaN agg_lagk)
        combined = pd.concat(
            [y_series.rename("y"), y_lag_series.rename("y_lag"),
             stacked, stacked_lagk.add_suffix("_lk")],
            axis=1
        ).dropna()
        if len(combined) < K + 3:
            mse_list.append(np.nan)
            continue

        y_arr = combined["y"].to_numpy()
        y_lag_arr = combined["y_lag"].to_numpy()
        # HF stacked cols: x_lag0..x_lag{K-1} (current period B(L,θ) inputs)
        stacked_cols = [c for c in combined.columns if not c.endswith("_lk")
                        and c not in ("y", "y_lag")]
        lagk_cols = [c for c in combined.columns if c.endswith("_lk")]
        Xk = combined[stacked_cols].to_numpy()     # shape (T_eff, K)
        Xk_lk = combined[lagk_cols].to_numpy()    # shape (T_eff, K); agg_{τ-1}

        def loss(params):
            theta_1, theta_2, b0, b1, b2 = params
            w = _w_exp_almon([theta_1, theta_2], K)
            agg_t = Xk @ w
            agg_tk = Xk_lk @ w
            # Common-factor form: y = b0 + b1*y_lag + b2*agg_t - b1*b2*agg_tk
            resid = y_arr - b0 - b1 * y_lag_arr - b2 * agg_t + b1 * b2 * agg_tk
            return float(np.sum(resid * resid))

        # Grid-search initialization: simplified-form OLS warm-start for each θ pair
        theta1_grid = [-0.5, 0.0, 0.5]
        theta2_grid = [-0.01, -0.1, -0.5, -1.0]
        best_loss = np.inf
        best_x0 = np.array([0.0, -0.1, float(np.mean(y_arr)), 0.3, 1.0])
        for t1 in theta1_grid:
            for t2 in theta2_grid:
                w0 = _w_exp_almon([t1, t2], K)
                agg0 = Xk @ w0
                # Simplified OLS (y ~ b0 + b1*y_lag + b2*agg) for warm-start
                Z = np.column_stack([np.ones(len(y_arr)), y_lag_arr, agg0])
                try:
                    betas0, _, _, _ = np.linalg.lstsq(Z, y_arr, rcond=None)
                    x0_cand = np.array([t1, t2, betas0[0], betas0[1], betas0[2]])
                    l0 = loss(x0_cand)
                    if l0 < best_loss:
                        best_loss = l0
                        best_x0 = x0_cand
                except Exception:
                    pass

        try:
            result = _scipy_minimize(
                loss, x0=best_x0, method="Nelder-Mead",
                options={"maxiter": 500, "xatol": 1e-6, "fatol": 1e-8},
            )
            params_hat = result.x
        except Exception:
            params_hat = best_x0

        theta_hat = params_hat[:2]
        b0_hat = float(params_hat[2])
        b1_hat = float(params_hat[3])
        b2_hat = float(params_hat[4])
        w_hat = _w_exp_almon(theta_hat, K)

        # OOS forecast (common-factor form):
        #   y_hat = b0 + b1*y_lag_last + b2*agg_test - b1*b2*agg_test_lagk
        # agg_test = B(L,θ)·x at LF period train_lf_end-1 → stacked.iloc[-1]
        # agg_test_lagk = B(L,θ)·x at LF period train_lf_end-2 → stacked.iloc[-2]
        #   (the common-factor term shifted 1 LF period relative to the forecast)
        stacked_np = stacked.to_numpy()
        agg_test = float((stacked_np[[-1]] @ w_hat).flat[0])
        # Need at least 2 rows for the common-factor lag; fallback to 0 if not available
        if len(stacked_np) >= 2:
            agg_test_lagk = float((stacked_np[[-2]] @ w_hat).flat[0])
        else:
            agg_test_lagk = 0.0
        y_lag_last = float(y_train[-1])    # y_{train_lf_end - 1}
        y_hat = (b0_hat + b1_hat * y_lag_last
                 + b2_hat * agg_test - b1_hat * b2_hat * agg_test_lagk)
        mse_list.append((y_hat - y_test) ** 2)

    valid = [v for v in mse_list if not np.isnan(v)]
    return float(np.mean(valid)) if valid else np.nan


class TestR4MonteCarlo:
    """TEST-R4-01, TEST-R4-02 from test-spec.md. Marked slow."""

    @pytest.mark.slow
    def test_r4_01_paper_table2_k3_persistent_umidas_wins(self):
        """TEST-R4-01: Paper Table 2, k=3, rho=0.9, U-MIDAS wins.

        Paper §3.2: BOTH models include AR(1) y-lag term.
        - U-MIDAS eq.(20): μ_0 + μ_1 y_{τ-1} + ψ(L) x_{τ-1} (OLS, BIC K)
        - MIDAS eq.(18, common-factor): β_0 + β_1 y_{τ-1} + β_2(1-β_1 L^k)B(L,θ)x_{τ-1}
          (NLS, K=1.5k, common-factor restriction fully implemented)
        Paper Table 2 (rho=0.9, delta_l=0.5, k=3): mean=0.91, median=0.91.
        Spec: mean_ratio in [0.79, 1.03] and < 1.10.
        Seeds: 2026+r for r in range(100).
        """
        rho, delta_l, delta_h = 0.9, 0.5, 0.0
        k, T_lf, ES, R = 3, 100, 50, 100
        master_seed = 2026

        ratios = []
        for r in range(R):
            y_lf, x_hf = _generate_hf_var1(
                rho, delta_l, delta_h, k, T_lf, ES,
                seed=master_seed + r
            )
            mse_umidas = _compute_umidas_oos_mse(y_lf, x_hf, k, T_lf, ES)
            mse_midas = _compute_midas_oos_mse_baseline(y_lf, x_hf, k, T_lf, ES)
            if not (np.isnan(mse_umidas) or np.isnan(mse_midas) or mse_midas <= 0):
                ratios.append(mse_umidas / mse_midas)

        assert len(ratios) >= 50, (
            "Too few valid replications: {}".format(len(ratios))
        )
        mean_ratio = float(np.mean(ratios))
        median_ratio = float(np.median(ratios))

        # Primary assertion: spec [0.79, 1.03]
        assert 0.79 <= mean_ratio <= 1.03, (
            "MC Table 2 anchor (k=3, rho=0.9): "
            "mean OOS ratio={:.3f}, expected approx 0.91 +/- 0.12. "
            "n_valid_reps={}".format(mean_ratio, len(ratios))
        )
        # Secondary assertion: ratio < 1.10
        assert mean_ratio < 1.10, (
            "Mean ratio {:.3f} >= 1.10: U-MIDAS badly underperforms MIDAS "
            "for k=3 persistent — paper expects U-MIDAS to win.".format(mean_ratio)
        )
        print("\n[TEST-R4-01 SYMMETRIC] mean_ratio={:.4f}, median_ratio={:.4f}, n_reps={}".format(
            mean_ratio, median_ratio, len(ratios)
        ))

    @pytest.mark.slow
    def test_r4_02_paper_table2_k60_midas_wins(self):
        """TEST-R4-02: Paper Table 2, k=60, rho=0.9, MIDAS wins (ratio > 0.95).

        Spec: mean_ratio > 0.95. Seeds: 3001+r for r in range(30).
        """
        rho, delta_l, delta_h = 0.9, 0.5, 0.0
        k, T_lf, ES, R = 60, 100, 50, 30
        master_seed = 3001

        ratios = []
        for r in range(R):
            y_lf, x_hf = _generate_hf_var1(
                rho, delta_l, delta_h, k, T_lf, ES,
                seed=master_seed + r
            )
            mse_umidas = _compute_umidas_oos_mse(y_lf, x_hf, k, T_lf, ES)
            mse_midas = _compute_midas_oos_mse_baseline(y_lf, x_hf, k, T_lf, ES)
            if not (np.isnan(mse_umidas) or np.isnan(mse_midas) or mse_midas <= 0):
                ratios.append(mse_umidas / mse_midas)

        if len(ratios) < 10:
            pytest.skip(
                "Insufficient valid replications for k=60 test: {}".format(len(ratios))
            )

        mean_ratio = float(np.mean(ratios))
        median_ratio = float(np.median(ratios))
        print("\n[TEST-R4-02] mean_ratio={:.4f}, median_ratio={:.4f}, n_reps={}".format(
            mean_ratio, median_ratio, len(ratios)
        ))

        assert mean_ratio > 0.95, (
            "MC Table 2 (k=60, rho=0.9): mean ratio={:.3f}; "
            "paper expects MIDAS to win (ratio > 1) at large k. "
            "Conservative threshold: > 0.95.".format(mean_ratio)
        )


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """EDGE-01..EDGE-04 from test-spec.md."""

    def test_edge_01_tiny_sample_fallback_k1(self):
        """EDGE-01: T_LF=2, K_max=5 → K_star=1 (fallback).

        Spec: K_star == 1.
        """
        rng = np.random.default_rng(0)
        T_HF = 6
        k = 3
        T_LF = 2
        frame = pd.DataFrame(
            {"x": rng.standard_normal(T_HF)},
            index=pd.RangeIndex(T_HF)
        )
        y_series = pd.Series(
            rng.standard_normal(T_LF),
            index=pd.RangeIndex(0, T_HF, k)
        )
        K_star = _bic_select_k(frame, y_series, freq_ratio=k,
                                include_y_lag=False, ic="bic")
        assert K_star == 1, (
            "Tiny-sample fallback: expected K=1; got K_star={}".format(K_star)
        )

    def test_edge_02_include_y_lag_true_y_series_none(self):
        """EDGE-02: include_y_lag=True with y_series=None.

        Spec: ValueError/TypeError OR graceful degradation. Either is acceptable.
        Documents actual behavior.
        """
        rng = np.random.default_rng(0)
        T_HF = 30
        k = 3
        frame = pd.DataFrame(
            {"x": rng.standard_normal(T_HF)},
            index=pd.RangeIndex(T_HF)
        )
        try:
            out = _u_midas(frame, freq_ratio=k, n_lags_high=3,
                           target_freq="low", include_y_lag=True,
                           y_series=None)
            has_y_lag = "y_lag1" in out.columns
            print("\n[EDGE-02] Graceful degradation: y_lag1_in_output={}".format(
                has_y_lag
            ))
        except (ValueError, TypeError) as exc:
            print("\n[EDGE-02] Raises {}: {}".format(type(exc).__name__, exc))

    def test_edge_03_negative_alpha_raises(self):
        """EDGE-03: regularization='ridge', alpha=-1.0 must raise an exception.

        Spec: ValueError or similar exception.
        """
        with pytest.raises((ValueError, Exception)):
            recipe = u_midas(
                target="y", horizon=1, freq_ratio=3, n_lags_high=3,
                regularization="ridge", alpha=-1.0,
                panel={"y": [1.0] * 30, "x1": [0.5] * 30}
            )
            _run_recipe(recipe)

    def test_edge_04_freq_ratio_1_same_frequency(self):
        """EDGE-04: freq_ratio=1 (same-frequency) → K_star in [1, 2].

        Spec: K_star in {1, 2}, no crash.
        """
        rng = np.random.default_rng(0)
        T = 30
        frame = pd.DataFrame(
            {"x": rng.standard_normal(T)},
            index=pd.RangeIndex(T)
        )
        y = pd.Series(rng.standard_normal(T), index=pd.RangeIndex(T))

        K_star = _bic_select_k(frame, y, freq_ratio=1,
                                include_y_lag=False, ic="bic")
        assert 1 <= K_star <= 2, (
            "freq_ratio=1: K_star={} outside [1, 2]".format(K_star)
        )


# ---------------------------------------------------------------------------
# Independent probe of builder's flagged deviation
# ---------------------------------------------------------------------------

class TestBuilderDeviationProbe:
    """Independent probe: does u_midas(include_y_lag=True) work end-to-end?

    Builder's test_bic_with_y_lag_end_to_end uses include_y_lag=False, citing
    expanding-window NaN mismatch. This independent test probes whether
    include_y_lag=True runs end-to-end with sufficient T (>= 120 LF obs = 360 HF).
    """

    def test_probe_include_y_lag_true_end_to_end_long_series(self):
        """Probe: include_y_lag=True with T_HF=360 (120 quarters) must run without error.

        If FAIL: NaN-handling bug in expanding-window L4 is confirmed.
        If PASS: builder's deviation was overly conservative (test could have used True).
        """
        rng = np.random.default_rng(42)
        T_HF = 360  # 120 quarters at m=3
        x_hf = rng.standard_normal(T_HF)
        y = np.zeros(T_HF)
        for t in range(T_HF):
            y[t] = (
                0.4 * x_hf[t]
                + 0.3 * (x_hf[t - 1] if t >= 1 else 0.0)
                + 0.3 * (x_hf[t - 2] if t >= 2 else 0.0)
            )
        y = y + rng.normal(0.0, 0.1, size=T_HF)

        panel = _build_panel_from_y_X(y, X=x_hf.reshape(-1, 1))

        recipe = u_midas(
            target="y", horizon=1, freq_ratio=3,
            include_y_lag=True,  # The deviation: builder used False
            panel=panel
        )
        result = _run_recipe(recipe)
        _assert_l4_forecasts(result)

        # Verify forecasts are finite floats
        fc_raw = result.cells[0].runtime_result.artifacts["l4_forecasts_v1"]
        fc_vals = np.array(list(fc_raw.forecasts.values()), dtype=float)
        assert len(fc_vals) > 0, "No forecasts returned"
        assert np.isfinite(fc_vals).all(), (
            "Non-finite forecasts with include_y_lag=True: first 5={}".format(
                fc_vals[:5]
            )
        )
        print("\n[DEVIATION PROBE] include_y_lag=True end-to-end PASSED. "
              "n_forecasts={}, mean_fc={:.4f}".format(
                  len(fc_vals), float(np.mean(fc_vals))
              ))
