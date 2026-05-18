"""Standalone function-op namespace.

Each export is a thin sklearn-style or pure-numeric wrapper around an
internal adapter, allowing partial use of macroforecast operations
without constructing a full recipe.

Cycle 22 POC: ``ridge_fit`` + ``theil_u1`` + ``theil_u2``.
Cycle 26: ``FitResultBase`` Protocol added.
Cycle 27: L5 metrics bulk standalone-ization (13 new ops).
Cycle 28: L4 linear family standalone-ization (7 ops).
Cycle 29: L6 statistical tests standalone-ization (7 ops).
Cycle 30: L3 basic panel transforms standalone-ization (10 ops).
Cycle 31: L3 advanced panel transforms standalone-ization (12 ops).
Cycle 32: L3 supervised/mixed transforms standalone-ization (6 ops).
Cycle 33: L3 final B1 transforms standalone-ization (8 ops).
Cycle 34: L2 clean panel ops standalone-ization (14 ops).
Cycle 35: L4 tree/ensemble family standalone-ization (6 ops).

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

    panel = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    print(mf.functions.diff_transform(panel))
    print(mf.functions.scale_transform(panel, method="zscore"))
    print(mf.functions.iqr_outlier_clean(panel, threshold=10.0))
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

from .transforms import (
    diff_transform,
    log_transform,
    log_diff_transform,
    pct_change_transform,
    cumsum_transform,
    ma_window_transform,
    lag_matrix,
    seasonal_lag_matrix,
    ma_increasing_order_transform,
    scale_transform,
    # Cycle 31: L3 advanced transforms
    hp_filter_transform,
    hamilton_filter_transform,
    savitzky_golay_transform,
    polynomial_expansion_transform,
    interaction_terms_transform,
    pca_transform,
    maf_per_variable_pca_transform,
    adaptive_ma_rf_transform,
    wavelet_transform,
    fourier_transform,
    asymmetric_trim_transform,
    season_dummy_transform,
    # Cycle 32: L3 supervised/mixed transforms
    scaled_pca_transform,
    supervised_pca_transform,
    partial_least_squares_transform,
    sliced_inverse_regression_transform,
    dfm_transform,
    feature_selection_transform,
    # Cycle 33: L3 final B1 transforms
    sparse_pca_transform,
    sparse_pca_chen_rohe_transform,
    varimax_transform,
    random_projection_transform,
    kernel_features_transform,
    nystroem_transform,
    time_trend_transform,
    holiday_transform,
)

# Cycle 34: L2 clean panel ops
from .clean import (
    iqr_outlier_clean,
    zscore_outlier_clean,
    winsorize_clean,
    em_factor_impute_clean,
    em_multivariate_impute_clean,
    mean_impute_clean,
    forward_fill_clean,
    linear_interpolate_clean,
    truncate_to_balanced_clean,
    drop_unbalanced_series_clean,
    zero_fill_leading_clean,
    apply_tcode_transform,
    freq_align_quarterly_to_monthly_clean,
    freq_align_monthly_to_quarterly_clean,
)

# Cycle 35: L4 tree/ensemble family
from .tree import (
    RandomForestFitResult, random_forest_fit,
    ExtraTreesFitResult, extra_trees_fit,
    GradientBoostingFitResult, gradient_boosting_fit,
    XGBoostFitResult, xgboost_fit,
    LightGBMFitResult, lightgbm_fit,
    CatBoostFitResult, catboost_fit,
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
    # Cycle 30: L3 basic panel transforms
    "diff_transform",
    "log_transform",
    "log_diff_transform",
    "pct_change_transform",
    "cumsum_transform",
    "ma_window_transform",
    "lag_matrix",
    "seasonal_lag_matrix",
    "ma_increasing_order_transform",
    "scale_transform",
    # Cycle 31: L3 advanced panel transforms
    "hp_filter_transform",
    "hamilton_filter_transform",
    "savitzky_golay_transform",
    "polynomial_expansion_transform",
    "interaction_terms_transform",
    "pca_transform",
    "maf_per_variable_pca_transform",
    "adaptive_ma_rf_transform",
    "wavelet_transform",
    "fourier_transform",
    "asymmetric_trim_transform",
    "season_dummy_transform",
    # Cycle 32: L3 supervised/mixed transforms
    "scaled_pca_transform",
    "supervised_pca_transform",
    "partial_least_squares_transform",
    "sliced_inverse_regression_transform",
    "dfm_transform",
    "feature_selection_transform",
    # Cycle 33: L3 final B1 transforms
    "sparse_pca_transform",
    "sparse_pca_chen_rohe_transform",
    "varimax_transform",
    "random_projection_transform",
    "kernel_features_transform",
    "nystroem_transform",
    "time_trend_transform",
    "holiday_transform",
    # Cycle 34: L2 clean panel ops
    "iqr_outlier_clean",
    "zscore_outlier_clean",
    "winsorize_clean",
    "em_factor_impute_clean",
    "em_multivariate_impute_clean",
    "mean_impute_clean",
    "forward_fill_clean",
    "linear_interpolate_clean",
    "truncate_to_balanced_clean",
    "drop_unbalanced_series_clean",
    "zero_fill_leading_clean",
    "apply_tcode_transform",
    "freq_align_quarterly_to_monthly_clean",
    "freq_align_monthly_to_quarterly_clean",
    # Cycle 35: L4 tree/ensemble family
    "RandomForestFitResult",
    "random_forest_fit",
    "ExtraTreesFitResult",
    "extra_trees_fit",
    "GradientBoostingFitResult",
    "gradient_boosting_fit",
    "XGBoostFitResult",
    "xgboost_fit",
    "LightGBMFitResult",
    "lightgbm_fit",
    "CatBoostFitResult",
    "catboost_fit",
]
