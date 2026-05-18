"""Standalone function-op namespace.

Each export is a thin sklearn-style or pure-numeric wrapper around an
internal adapter, allowing partial use of macroforecast operations
without constructing a full recipe.

Cycle 22 POC: ``ridge_fit`` + ``theil_u1`` + ``theil_u2``.
Cycle 26: ``FitResultBase`` Protocol added.
Cycle 27: L5 metrics bulk standalone-ization (13 new ops).
Cycle 28: L4 linear family standalone-ization (7 ops).
Cycle 29: L6 statistical tests standalone-ization (7 ops).

Example usage::

    import macroforecast as mf
    import numpy as np
    import pandas as pd

    X = pd.DataFrame({"x1": [1, 2, 3, 4, 5], "x2": [2, 3, 4, 5, 6]})
    y = pd.Series([2.0, 3.0, 4.0, 5.0, 6.0])

    result = mf.functions.ridge_fit(X, y, alpha=0.5)
    print(result.coef_)          # array of coefficients
    print(result.predict(X))     # predictions
    print(result.summary())      # statsmodels-style text table

    u1 = mf.functions.theil_u1(np.array([1, 2, 3]), np.array([1.5, 2.5, 3.5]))
    print(u1)                    # 0.1044...

    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.1, 3.1])
    print(mf.functions.mse(y_true, y_pred))   # 0.01
    print(mf.functions.rmse(y_true, y_pred))  # 0.1
    print(mf.functions.mae(y_true, y_pred))   # 0.1
"""
from __future__ import annotations

from ._base import FitResultBase
from .ridge import RidgeFitResult, ridge_fit
from .theil_u import theil_u1, theil_u2
from .metrics import (
    mse,
    rmse,
    mae,
    medae,
    mape,
    relative_mse,
    relative_mae,
    mse_reduction,
    r2_oos,
    interval_score,
    coverage_rate,
    success_ratio,
    pesaran_timmermann_metric,
)
from .linear import (
    OLSFitResult, ols_fit,
    LassoFitResult, lasso_fit,
    ElasticNetFitResult, elastic_net_fit,
    LassoPathFitResult, lasso_path_fit,
    BayesianRidgeFitResult, bayesian_ridge_fit,
    HuberFitResult, huber_fit,
    GLMBoostFitResult, glmboost_fit,
)

from .tests import (
    DMTestResult, dm_test,
    GWTestResult, gw_test,
    DMPTestResult, dmp_test,
    HNTestResult, hn_test,
    CWTestResult, cw_test,
    EncNewTestResult, enc_new_test,
    EncTTestResult, enc_t_test,
)

__all__ = [
    "FitResultBase",
    "RidgeFitResult",
    "ridge_fit",
    "theil_u1",
    "theil_u2",
    "mse",
    "rmse",
    "mae",
    "medae",
    "mape",
    "relative_mse",
    "relative_mae",
    "mse_reduction",
    "r2_oos",
    "interval_score",
    "coverage_rate",
    "success_ratio",
    "pesaran_timmermann_metric",
    "OLSFitResult",
    "ols_fit",
    "LassoFitResult",
    "lasso_fit",
    "ElasticNetFitResult",
    "elastic_net_fit",
    "LassoPathFitResult",
    "lasso_path_fit",
    "BayesianRidgeFitResult",
    "bayesian_ridge_fit",
    "HuberFitResult",
    "huber_fit",
    "GLMBoostFitResult",
    "glmboost_fit",
    "DMTestResult",
    "dm_test",
    "GWTestResult",
    "gw_test",
    "DMPTestResult",
    "dmp_test",
    "HNTestResult",
    "hn_test",
    "CWTestResult",
    "cw_test",
    "EncNewTestResult",
    "enc_new_test",
    "EncTTestResult",
    "enc_t_test",
]
