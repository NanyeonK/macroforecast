"""Newey-West HAC option for the predictor-screen t-statistic.

Covers the additive ``hac_lags`` knob on :func:`marginal_t_stats` and
:func:`fit_predictor_screen`:

* the default (``hac_lags=None``) path is bit-identical to the pre-change
  homoskedastic OLS estimator;
* a positive ``hac_lags`` reproduces statsmodels' Bartlett-kernel HAC
  (``cov_type='HAC'``, ``use_correction=False``) to machine precision;
* screening an autocorrelated target with HAC selects a (strict) subset of the
  OLS selection, because the corrected standard errors shrink the inflated
  t-stats.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macroforecast.feature_engineering.screening import (
    _regression_design,
    fit_predictor_screen,
    marginal_t_stats,
)


def _ar1(n: int, rho: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    innovations = rng.normal(scale=sigma, size=n)
    series = np.empty(n)
    series[0] = innovations[0]
    for t in range(1, n):
        series[t] = rho * series[t - 1] + innovations[t]
    return series


def _reference_ols_t_stats(
    X: np.ndarray, y: np.ndarray, controls: np.ndarray | None = None
) -> np.ndarray:
    """Verbatim copy of the pre-change homoskedastic OLS t-stat estimator."""

    x_values = np.asarray(X, dtype=float)
    y_values = np.asarray(y, dtype=float).reshape(-1)
    control_values = (
        np.empty((len(y_values), 0), dtype=float)
        if controls is None
        else np.asarray(controls, dtype=float)
    )
    n_rows = x_values.shape[0]
    out = np.zeros(x_values.shape[1], dtype=float)
    if n_rows <= control_values.shape[1] + 2:
        return out
    for idx in range(x_values.shape[1]):
        design = _regression_design(control_values, x_values[:, [idx]])
        coef = np.linalg.pinv(design) @ y_values
        resid = y_values - design @ coef
        dof = max(n_rows - design.shape[1], 1)
        sigma2 = float(resid @ resid) / dof
        cov = sigma2 * np.linalg.pinv(design.T @ design)
        se = float(np.sqrt(max(cov[-1, -1], 0.0)))
        out[idx] = 0.0 if se <= 1e-12 else float(coef[-1] / se)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def test_marginal_t_stats_hac_none_is_bit_identical_to_ols() -> None:
    """Golden identity: ``hac_lags=None`` reproduces the OLS estimator exactly."""

    rng = np.random.default_rng(12345)
    n = 120
    X = np.column_stack(
        [_ar1(n, 0.5, 1.0, rng) for _ in range(4)]
    )
    controls = rng.normal(size=(n, 2))
    y = 0.4 * X[:, 0] - 0.25 * X[:, 2] + rng.normal(size=n)

    reference = _reference_ols_t_stats(X, y, controls=controls)
    default_call = marginal_t_stats(X, y, controls=controls)
    explicit_none = marginal_t_stats(X, y, controls=controls, hac_lags=None)

    # Bit-identical: no tolerance.
    assert np.array_equal(default_call, reference)
    assert np.array_equal(explicit_none, reference)


@pytest.mark.parametrize("hac_lags", [0, 1, 5, 12])
def test_marginal_t_stats_hac_matches_statsmodels(hac_lags: int) -> None:
    """HAC t-stat matches statsmodels ``cov_type='HAC'`` (no small-sample corr.)."""

    sm = pytest.importorskip("statsmodels.api")

    rng = np.random.default_rng(20260714)
    n = 240
    x = _ar1(n, 0.6, 1.0, rng)
    resid = _ar1(n, 0.7, 1.0, rng)  # autocorrelated errors -> HAC diverges from OLS
    y = 0.3 * x + resid

    ours = marginal_t_stats(x.reshape(-1, 1), y, hac_lags=hac_lags)[0]
    design = sm.add_constant(x)
    sm_t = (
        sm.OLS(y, design)
        .fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags, "use_correction": False})
        .tvalues[1]
    )
    assert abs(ours - sm_t) < 1e-6

    # And it genuinely differs from the (inflated) homoskedastic OLS t-stat.
    ols_t = marginal_t_stats(x.reshape(-1, 1), y, hac_lags=None)[0]
    if hac_lags > 0:
        assert abs(ours) < abs(ols_t)


def test_marginal_t_stats_hac_rejects_negative_lags() -> None:
    x = np.random.default_rng(0).normal(size=(50, 1))
    y = np.random.default_rng(1).normal(size=50)
    with pytest.raises(ValueError):
        marginal_t_stats(x, y, hac_lags=-1)


def test_predictor_screen_hac_selects_fewer_predictors_on_autocorrelated_target() -> None:
    """Sanity: on a persistent target, HAC screens a strict subset of OLS picks."""

    rng = np.random.default_rng(42)
    n = 200
    target = pd.Series(_ar1(n, 0.85, 1.0, rng), name="tgt")
    source = pd.DataFrame(
        {f"x{j}": _ar1(n, 0.85, 1.0, rng) for j in range(12)}
    )
    threshold = 1.96

    ols = fit_predictor_screen(
        source, target, method="t_stat", threshold=threshold, hac_lags=None
    )
    hac = fit_predictor_screen(
        source, target, method="t_stat", threshold=threshold, hac_lags=8
    )

    ols_selected = set(ols.selected_columns)
    hac_selected = set(hac.selected_columns)

    # Fewer-or-equal is the guaranteed sanity; here it is strictly fewer.
    assert len(hac_selected) <= len(ols_selected)
    assert len(hac_selected) < len(ols_selected)
    assert hac_selected <= ols_selected
    # HAC lag choice is recorded in the fitted metadata; OLS default omits it.
    assert hac.metadata.get("hac_lags") == 8
    assert "hac_lags" not in ols.metadata
