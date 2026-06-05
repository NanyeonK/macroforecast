"""Regression tests for two parity bugs vs reference packages."""
from __future__ import annotations

import numpy as np
import pandas as pd

from macroforecast.models.timeseries import _prior_scale_matrix
from macroforecast.interpretation.core import _InternalVARResults


def test_prior_scale_matrix_scalar_is_isotropic_not_rank1():
    k = 3
    out = _prior_scale_matrix(0.5, k)
    # Must be the positive-definite isotropic scale 0.5*I, not a rank-1 full matrix.
    np.testing.assert_allclose(out, 0.5 * np.eye(k))
    eig = np.linalg.eigvalsh(out)
    assert (eig > 1e-12).all(), eig  # positive definite (old code gave [0,0,1.5])


class _FakeVAR:
    """Minimal estimator exposing the attributes _InternalVARResults reads."""

    def __init__(self, names, n_lag, coef, datamat):
        self.names_ = names
        self.n_lag = n_lag
        self.coef_ = coef
        self.datamat_ = datamat


def test_internal_var_sigma_u_uses_regressor_df():
    rng = np.random.default_rng(0)
    k, p, n_det = 2, 2, 1
    n_reg = k * p + n_det  # 5 regressors per equation
    T_eff = 20
    y = rng.normal(size=(T_eff, k))
    X = rng.normal(size=(T_eff, n_reg))
    coef = np.zeros((k, n_reg))  # resid == y exactly
    datamat = pd.DataFrame(np.hstack([y, X]))
    res = _InternalVARResults(_FakeVAR(("a", "b"), p, coef, datamat))

    from macroforecast.interpretation.core import _positive_definite_covariance
    sse = y.T @ y
    expected = _positive_definite_covariance(sse / (T_eff - n_reg))  # df = 15
    wrong = _positive_definite_covariance(sse / (T_eff - k))         # old df = 18
    np.testing.assert_allclose(res.sigma_u, expected, rtol=1e-9, atol=1e-9)
    assert not np.allclose(res.sigma_u, wrong)
