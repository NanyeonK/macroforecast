"""Tests for the four L7 ops re-promoted in v0.2 alongside the ALE PR:

* #189: ``fevd`` / ``historical_decomposition`` / ``generalized_irf``
  (orthogonalised IRFs via statsmodels VAR results object)
* #190: ``mrf_gtvp`` (time-varying coefficient series from the GTVP
  betas surfaced by ``_MRFExternalWrapper`` — re-anchored to the
  vendored ``_vendor/macro_random_forest`` reference implementation in
  v0.8.9)
* #191: ``lasso_inclusion_frequency`` (bootstrap-resample inclusion
  frequency)
* #193: ``friedman_h_interaction`` (Friedman & Popescu 2008 H statistic)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression

from macroforecast.core.runtime import (
    _MRFExternalWrapper,
    _friedman_h_table,
    _lasso_inclusion_frame,
    _mrf_gtvp_coefficient_frame,
    _var_impulse_frame,
)
from macroforecast.core.types import ModelArtifact
from macroforecast.core.ops.l7_ops import HONESTY_DEMOTED_L7_OPS


# ---------------------------------------------------------------------------
# #193 Friedman H
# ---------------------------------------------------------------------------

def test_friedman_h_returns_pairwise_statistic_per_pair():
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = pd.Series(X["a"] * X["b"] + rng.normal(scale=0.1, size=n))  # interaction!
    fitted = LinearRegression().fit(X, y)
    artifact = ModelArtifact(
        model_id="m",
        family="ols",
        fitted_object=fitted,
        framework="sklearn",
        feature_names=tuple(X.columns),
    )
    frame = _friedman_h_table(artifact, X, n_grid=4)
    assert {"feature", "importance"}.issubset(frame.columns)
    # 3 features choose 2 = 3 pairs.
    assert len(frame) == 3
    # H is in [0, 1].
    assert frame["importance"].between(0, 1).all()


# ---------------------------------------------------------------------------
# #191 lasso_inclusion_frequency
# ---------------------------------------------------------------------------

def test_lasso_inclusion_frequency_resamples_when_panel_supplied():
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=list("abcde"))
    y = pd.Series(2.0 * X["a"] - 0.5 * X["b"] + rng.normal(scale=0.5, size=n))
    fitted = Lasso(alpha=0.05, max_iter=20000).fit(X, y)
    artifact = ModelArtifact(
        model_id="m",
        family="lasso",
        fitted_object=fitted,
        framework="sklearn",
        feature_names=tuple(X.columns),
    )
    frame = _lasso_inclusion_frame(artifact, X=X, y=y, n_bootstraps=10, seed=0)
    assert "n_bootstraps_run" in frame.columns
    # Bootstrap inclusion frequency must lie in [0, 1].
    assert frame["importance"].between(0, 1).all()
    # The features actually driving y should have higher inclusion than the
    # noise features.
    importance = frame.set_index("feature")["importance"]
    assert importance["a"] > 0.5  # informative
    assert importance["a"] >= importance["c"]


def test_lasso_inclusion_falls_back_when_panel_missing():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=list("abc"))
    y = pd.Series(rng.normal(size=40))
    fitted = Lasso(alpha=0.1).fit(X, y)
    artifact = ModelArtifact(
        model_id="m",
        family="lasso",
        fitted_object=fitted,
        framework="sklearn",
        feature_names=tuple(X.columns),
    )
    frame = _lasso_inclusion_frame(artifact, X=None, y=None)
    # Binary single-fit indicator (0 or 1).
    assert "n_bootstraps_run" not in frame.columns
    assert set(frame["importance"].unique()).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# #190 mrf_gtvp
# ---------------------------------------------------------------------------

def test_mrf_gtvp_produces_per_row_coefficient_path():
    """L7 ``mrf_gtvp`` reads the GTVP β̂(t) series directly from the
    ``_MRFExternalWrapper._cached_betas`` populated during predict.
    ``coefficient_path`` length therefore equals the *full panel*
    size (n_train + n_test) used for the most recent forecast."""

    rng = np.random.default_rng(0)
    n_train, n_test = 50, 10
    n = n_train + n_test
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    y = pd.Series(0.5 * X["a"] + rng.normal(scale=0.1, size=n))
    mrf = _MRFExternalWrapper(B=8, parallelise=False, n_cores=1, random_state=0)
    mrf.fit(X.iloc[:n_train], y.iloc[:n_train])
    mrf.predict(X.iloc[n_train:])
    artifact = ModelArtifact(
        model_id="m",
        family="macroeconomic_random_forest",
        fitted_object=mrf,
        framework="sklearn",
        feature_names=tuple(X.columns),
    )
    frame = _mrf_gtvp_coefficient_frame(artifact, X)
    assert "coefficient_path" in frame.columns
    for _, row in frame.iterrows():
        assert len(row["coefficient_path"]) == n
    # Importance: time-average of |β̂(t)| (nan-safe; mrf-web leaves OOS rows
    # NaN because they're not covered by the in-sample bootstrap).
    assert (frame["importance"] >= 0).all()


# ---------------------------------------------------------------------------
# #189 IRF / FEVD / HD via statsmodels VAR
# ---------------------------------------------------------------------------

def test_var_impulse_frame_status_falls_back_for_non_var_model():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=list("abc"))
    y = pd.Series(rng.normal(size=40))
    fitted = LinearRegression().fit(X, y)
    artifact = ModelArtifact(
        model_id="m",
        family="ols",
        fitted_object=fitted,
        framework="sklearn",
        feature_names=tuple(X.columns),
    )
    # ``generalized_irf`` is now future-gated (v0.8.9 honesty pass V2.3);
    # the operational Cholesky variant is ``orthogonalised_irf``.
    for op_name in ("fevd", "historical_decomposition", "orthogonalised_irf"):
        frame = _var_impulse_frame(artifact, op_name=op_name)
        assert "status" in frame.columns
        # Non-VAR fall-through is documented.
        assert "fallback" in frame["status"].iloc[0]


# ---------------------------------------------------------------------------
# Promotion pin
# ---------------------------------------------------------------------------

def test_no_l7_ops_remain_in_honesty_demoted_bucket():
    """After #194 lands the gradient family the bucket is empty -- every
    L7 honesty-pass demotion has a real runtime."""

    assert HONESTY_DEMOTED_L7_OPS == ()
