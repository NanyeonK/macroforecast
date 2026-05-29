"""Standalone function-op namespace.

Each export is a thin sklearn-style or pure-numeric wrapper around an
internal adapter, allowing partial use of macroforecast operations
without constructing a full recipe.

v0.1.0: ``ridge_fit`` + ``theil_u1`` + ``theil_u2``.
v0.1.0: ``FitResultBase`` Protocol added.
L5 metrics bulk standalone-ization (v0.1.0, 13 ops).
L4 linear family standalone-ization (v0.1.0, 7 ops).
L6 statistical tests standalone-ization (v0.1.0, 7 ops).
L3 basic panel transforms standalone-ization (v0.8.0, 10 ops).
L3 advanced panel transforms standalone-ization (v0.8.0, 12 ops).
L3 supervised/mixed transforms standalone-ization (v0.8.0, 6 ops).
L3 final B1 transforms standalone-ization (v0.8.0, 8 ops).
Preprocessing clean-panel ops standalone-ization (v0.8.0, 14 ops).
L4 tree/ensemble family standalone-ization (v0.8.0, 6 ops).
L7 importance standalone callables (v0.8.0, 8 ops).

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
    # L3 advanced transforms (v0.8.0)
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
    # L3 supervised/mixed transforms (v0.8.0)
    scaled_pca_transform,
    supervised_pca_transform,
    partial_least_squares_transform,
    sliced_inverse_regression_transform,
    dfm_transform,
    feature_selection_transform,
    # L3 final B1 transforms (v0.8.0)
    sparse_pca_transform,
    sparse_pca_chen_rohe_transform,
    varimax_transform,
    random_projection_transform,
    kernel_features_transform,
    nystroem_transform,
    time_trend_transform,
    holiday_transform,
)

# Preprocessing clean-panel ops (v0.8.0)
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

# L4 tree/ensemble family (v0.8.0)
from .tree import (
    RandomForestFitResult, random_forest_fit,
    ExtraTreesFitResult, extra_trees_fit,
    GradientBoostingFitResult, gradient_boosting_fit,
    XGBoostFitResult, xgboost_fit,
    LightGBMFitResult, lightgbm_fit,
    CatBoostFitResult, catboost_fit,
    # tree/ensemble gap callables (v0.9.5)
    SlowGrowingTreeFitResult, slow_growing_tree_fit,
    QuantileRegressionForestFitResult, quantile_regression_forest_fit,
    BaggingFitResult, bagging_fit,
    BoogingFitResult, booging_fit,
    MacroRandomForestFitResult, macro_random_forest_fit,
)

# L4 deep family (v0.8.0)
from .deep import (
    MLPFitResult, mlp_fit,
    LSTMFitResult, lstm_fit,
    GRUFitResult, gru_fit,
    TransformerFitResult, transformer_fit,
    # neural gap callable (v0.9.5)
    HemisphereNNFitResult, hemisphere_nn_fit,
)

# L4 timeseries family (v0.8.0)
from .timeseries import (
    GARCHFitResult,
    VARFitResult, var_fit,
    BVARMinnesotaFitResult, bvar_minnesota_fit,
    BVARNIWFitResult, bvar_niw_fit,
    ARFitResult, ar_fit,
    FARFitResult, far_fit,
    PCRFitResult, pcr_fit,
    FAVARFitResult, favar_fit,
    GARCH11FitResult, garch11_fit,
    EGARCHFitResult, egarch_fit,
    RealizedGARCHFitResult, realized_garch_fit,
    ETSFitResult, ets_fit,
    ThetaFitResult, theta_fit,
    HoltWintersFitResult, holt_winters_fit,
    DFMFitResult, dfm_fit,
)

# L4 misc family (v0.8.0)
from .misc import (
    SVRFitResult,
    SVRLinearFitResult, svr_linear_fit,
    SVRRBFFitResult, svr_rbf_fit,
    SVRPolyFitResult, svr_poly_fit,
    KNNFitResult, knn_fit,
    KernelRidgeFitResult, kernel_ridge_fit,
    MARSFitResult, mars_fit,
)

# L4 MIDAS family standalone callables (v0.9.5, 4 ops)
from .midas import (
    MidasFitResult,
    midas_almon_fit,
    midas_beta_fit,
    midas_step_fit,
    unrestricted_midas_fit,
)

# L4 ridge-variant family standalone callables (v0.9.5, 4 ops)
from .ridge_variants import (
    nonneg_ridge_fit,
    random_walk_ridge_fit,
    shrink_to_target_ridge_fit,
    fused_difference_ridge_fit,
)

# L7 importance standalone callables (v0.8.0)
from .importance import (
    NativeImportanceResult,
    PermutationImportanceResult,
    CondPermutationImportanceResult,
    PDPImportanceResult,
    ALEImportanceResult,
    SHAPImportanceResult,
    model_native_linear_coef_importance,
    model_native_tree_importance,
    permutation_importance,
    cond_permutation_importance,
    partial_dependence_importance,
    ale_importance,
    shap_tree_importance,
    shap_linear_importance,
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
    # L3 basic panel transforms (v0.8.0)
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
    # L3 advanced panel transforms (v0.8.0)
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
    # L3 supervised/mixed transforms (v0.8.0)
    "scaled_pca_transform",
    "supervised_pca_transform",
    "partial_least_squares_transform",
    "sliced_inverse_regression_transform",
    "dfm_transform",
    "feature_selection_transform",
    # L3 final B1 transforms (v0.8.0)
    "sparse_pca_transform",
    "sparse_pca_chen_rohe_transform",
    "varimax_transform",
    "random_projection_transform",
    "kernel_features_transform",
    "nystroem_transform",
    "time_trend_transform",
    "holiday_transform",
    # Preprocessing clean-panel ops (v0.8.0)
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
    # L4 tree/ensemble family (v0.8.0)
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
    # L4 deep family (v0.8.0)
    "MLPFitResult",
    "mlp_fit",
    "LSTMFitResult",
    "lstm_fit",
    "GRUFitResult",
    "gru_fit",
    "TransformerFitResult",
    "transformer_fit",
    # tree/ensemble gap callables (v0.9.5)
    "SlowGrowingTreeFitResult",
    "slow_growing_tree_fit",
    "QuantileRegressionForestFitResult",
    "quantile_regression_forest_fit",
    "BaggingFitResult",
    "bagging_fit",
    "BoogingFitResult",
    "booging_fit",
    "MacroRandomForestFitResult",
    "macro_random_forest_fit",
    # neural gap callable (v0.9.5)
    "HemisphereNNFitResult",
    "hemisphere_nn_fit",
    # L4 timeseries family (v0.8.0)
    "GARCHFitResult",
    "VARFitResult",
    "var_fit",
    "BVARMinnesotaFitResult",
    "bvar_minnesota_fit",
    "BVARNIWFitResult",
    "bvar_niw_fit",
    "ARFitResult",
    "ar_fit",
    "FARFitResult",
    "far_fit",
    "PCRFitResult",
    "pcr_fit",
    "FAVARFitResult",
    "favar_fit",
    "GARCH11FitResult",
    "garch11_fit",
    "EGARCHFitResult",
    "egarch_fit",
    "RealizedGARCHFitResult",
    "realized_garch_fit",
    "ETSFitResult",
    "ets_fit",
    "ThetaFitResult",
    "theta_fit",
    "HoltWintersFitResult",
    "holt_winters_fit",
    "DFMFitResult",
    "dfm_fit",
    # L4 misc family (v0.8.0)
    "SVRFitResult",
    "SVRLinearFitResult",
    "svr_linear_fit",
    "SVRRBFFitResult",
    "svr_rbf_fit",
    "SVRPolyFitResult",
    "svr_poly_fit",
    "KNNFitResult",
    "knn_fit",
    "KernelRidgeFitResult",
    "kernel_ridge_fit",
    "MARSFitResult",
    "mars_fit",
    # L7 importance standalone callables (v0.8.0)
    "NativeImportanceResult",
    "PermutationImportanceResult",
    "CondPermutationImportanceResult",
    "PDPImportanceResult",
    "ALEImportanceResult",
    "SHAPImportanceResult",
    "model_native_linear_coef_importance",
    "model_native_tree_importance",
    "permutation_importance",
    "cond_permutation_importance",
    "partial_dependence_importance",
    "ale_importance",
    "shap_tree_importance",
    "shap_linear_importance",
    # L4 MIDAS family (v0.9.5)
    "MidasFitResult",
    "midas_almon_fit",
    "midas_beta_fit",
    "midas_step_fit",
    "unrestricted_midas_fit",
    # L4 ridge-variant family (v0.9.5)
    "nonneg_ridge_fit",
    "random_walk_ridge_fit",
    "shrink_to_target_ridge_fit",
    "fused_difference_ridge_fit",
]