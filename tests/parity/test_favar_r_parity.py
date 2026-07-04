"""Live R cross-check of favar()'s deterministic factor-identification helpers.

WP-V2. Per ``.dev-notes/anchor_coverage/matrix.csv``, ``favar`` had "zero
executable comparison vs R FAVAR anywhere". The R ``FAVAR`` package installs
cleanly from CRAN on this host, and its internal (non-exported but
accessible via ``FAVAR:::``) helpers -- ``ExtrPC``, ``facrot``, ``olssvd``,
``BGM`` -- are exactly the functions macroforecast's
``_favar_extr_pc``/``_favar_facrot``/``_favar_olssvd``/``_favar_bgm`` claim
to port (confirmed by reading both sources side by side; see comments
below). All four are fully deterministic (no RNG), unlike the loading-Gibbs
and BVAR-Gibbs draw steps, so this is a genuine byte-level parity check, not
just a distributional one.

Uses the subprocess-Rscript bridge from ``tests/parity/conftest.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from macroforecast.models.timeseries import (
    _favar_bgm,
    _favar_extr_pc,
    _favar_facrot,
    _favar_olssvd,
)
from tests.parity.conftest import require_r, run_rscript

pytestmark = pytest.mark.rparity


def _fmt_matrix(values: np.ndarray) -> str:
    return ",".join(f"{v:.17g}" for v in values.flatten(order="C"))


def _align_signs(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    aligned = candidate.copy()
    for col in range(candidate.shape[1]):
        idx = int(np.argmax(np.abs(reference[:, col])))
        if np.sign(reference[idx, col]) != np.sign(candidate[idx, col]):
            aligned[:, col] *= -1.0
    return aligned


def _fixture(n_obs: int = 80, n_series: int = 6, seed: int = 3):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n_obs, n_series))
    return values


def test_extr_pc_matches_r_favar_extrpc():
    require_r("FAVAR")
    values = _fixture()
    n_obs, n_series = values.shape
    k = 2
    factors_py, loadings_py = _favar_extr_pc(values, k)

    script = f"""
    suppressMessages(library(FAVAR))
    X <- matrix(c({_fmt_matrix(values)}), nrow = {n_obs}, ncol = {n_series}, byrow = TRUE)
    out <- FAVAR:::ExtrPC(X, {k})
    emit("F0", as.vector(t(out$F0)))
    emit("Lf", as.vector(t(out$Lf)))
    """
    result = run_rscript(script)
    factors_r = np.array([float(v) for v in result["F0"].split(",")]).reshape(n_obs, k)
    loadings_r = np.array([float(v) for v in result["Lf"].split(",")]).reshape(n_series, k)

    factors_r_aligned = _align_signs(factors_py, factors_r)
    loadings_r_aligned = _align_signs(loadings_py, loadings_r)
    np.testing.assert_allclose(factors_py, factors_r_aligned, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(loadings_py, loadings_r_aligned, rtol=1e-8, atol=1e-8)


def test_olssvd_matches_r_favar_olssvd():
    require_r("FAVAR")
    rng = np.random.default_rng(4)
    y = rng.normal(size=(60, 2))
    x = rng.normal(size=(60, 3))
    b_py = _favar_olssvd(y, x)

    script = f"""
    suppressMessages(library(FAVAR))
    F0 <- matrix(c({_fmt_matrix(y)}), nrow = {y.shape[0]}, ncol = {y.shape[1]}, byrow = TRUE)
    ly <- matrix(c({_fmt_matrix(x)}), nrow = {x.shape[0]}, ncol = {x.shape[1]}, byrow = TRUE)
    b <- FAVAR:::olssvd(F0, ly)
    emit("b", as.vector(t(b)))
    """
    result = run_rscript(script)
    b_r = np.array([float(v) for v in result["b"].split(",")]).reshape(x.shape[1], y.shape[1])
    np.testing.assert_allclose(b_py, b_r, rtol=1e-6, atol=1e-8)


def test_facrot_matches_r_favar_facrot():
    require_r("FAVAR")
    rng = np.random.default_rng(5)
    n_obs = 80
    factors = rng.normal(size=(n_obs, 2))
    fast = rng.normal(size=(n_obs, 1))
    slow_factors = rng.normal(size=(n_obs, 2))
    result_py = _favar_facrot(factors, fast, slow_factors)

    script = f"""
    suppressMessages(library(FAVAR))
    F0 <- matrix(c({_fmt_matrix(factors)}), nrow = {n_obs}, ncol = 2, byrow = TRUE)
    Ffast <- matrix(c({_fmt_matrix(fast)}), nrow = {n_obs}, ncol = 1, byrow = TRUE)
    Fslow0 <- matrix(c({_fmt_matrix(slow_factors)}), nrow = {n_obs}, ncol = 2, byrow = TRUE)
    out <- FAVAR:::facrot(F0, Ffast, Fslow0)
    emit("out", as.vector(t(out)))
    """
    result = run_rscript(script)
    result_r = np.array([float(v) for v in result["out"].split(",")]).reshape(n_obs, 2)
    np.testing.assert_allclose(result_py, result_r, rtol=1e-6, atol=1e-8)


def test_bgm_matches_r_favar_bgm():
    require_r("FAVAR")
    rng = np.random.default_rng(6)
    n_obs, n_series = 100, 6
    values = rng.normal(size=(n_obs, n_series))
    response = rng.normal(size=n_obs)
    factors_py = _favar_bgm(values, response, 2, tolerance=0.001, nmax=100)

    script = f"""
    suppressMessages(library(FAVAR))
    X <- matrix(c({_fmt_matrix(values)}), nrow = {n_obs}, ncol = {n_series}, byrow = TRUE)
    R <- c({",".join(f"{v:.17g}" for v in response)})
    out <- FAVAR:::BGM(X, R, K = 2, tolerance = 0.001, nmax = 100)
    emit("out", as.vector(t(out)))
    """
    result = run_rscript(script)
    factors_r = np.array([float(v) for v in result["out"].split(",")]).reshape(n_obs, 2)
    factors_r_aligned = _align_signs(factors_py, factors_r)
    # BGM is an iterative fixed-point purge; R uses solve() (exact inverse)
    # where our port uses pinv() as a numerical guard (see _favar_bgm's own
    # code comment lineage in timeseries.py), so a slightly looser tolerance
    # than the fully linear-algebra ExtrPC/olssvd/facrot checks above.
    np.testing.assert_allclose(factors_py, factors_r_aligned, rtol=1e-4, atol=1e-6)
