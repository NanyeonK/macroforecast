# Model x Forecast Policy Matrix

[Back to Models and Features](model_overview.md)

This page states which forecast policies are statistically supported for each
registered model. It is generated from `mf.list_model_specs()`, each model's
`ModelSpec.default_params`, and the package's direct-policy guard.

Status meanings:

- `supported`: the runner policy applies to this model family.
- `supported-via-direct-projection`: the model has an explicit validated direct
  projection mode, used for `direct` and `direct_average`.
- `guarded-unsupported`: the combination is blocked by default because the model
  forecasts by iterating its own dynamics rather than by fitting an h-step direct
  projection.
- `unsupported`: the runner does not define this policy for the model's input
  contract.

The measured-scan column is only a point-in-time runtime smoke scan. `OK` means a
combination ran, not that its forecast object has the right statistical meaning.

| Model | Family | Input | direct | direct_average | path_average | recursive | Measured scan |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `albacore_components` | assemblage | `supervised` | supported | supported | supported | supported | not available |
| `albacore_ranks` | assemblage | `supervised` | supported | supported | supported | supported | not available |
| `assemblage_regression` | assemblage | `supervised` | supported | supported | supported | supported | not available |
| `component_aggregation` | assemblage | `supervised` | supported | supported | supported | supported | not available |
| `rank_aggregation` | assemblage | `supervised` | supported | supported | supported | supported | not available |
| `supervised_aggregation` | assemblage | `supervised` | supported | supported | supported | supported | not available |
| `pls` | composite | `supervised` | supported | supported | supported | supported | not available |
| `scaled_pca` | composite | `supervised` | supported | supported | supported | supported | not available |
| `supervised_pca` | composite | `supervised` | supported | supported | supported | supported | not available |
| `supervised_scaled_pca` | composite | `supervised` | supported | supported | supported | supported | not available |
| `far` | factor | `supervised` | supported-via-direct-projection | supported-via-direct-projection | supported | supported | not available |
| `favar` | factor | `supervised` | guarded-unsupported | guarded-unsupported | supported | supported | not available |
| `adaptive_elastic_net` | linear | `supervised` | supported | supported | supported | supported | not available |
| `adaptive_lasso` | linear | `supervised` | supported | supported | supported | supported | not available |
| `bayesian_ridge` | linear | `supervised` | supported | supported | supported | supported | not available |
| `elastic_net` | linear | `supervised` | supported | supported | supported | supported | not available |
| `fused_difference_ridge` | linear | `supervised` | supported | supported | supported | supported | not available |
| `glmboost` | linear | `supervised` | supported | supported | supported | supported | not available |
| `group_lasso` | linear | `supervised` | supported | supported | supported | supported | not available |
| `huber` | linear | `supervised` | supported | supported | supported | supported | not available |
| `lasso` | linear | `supervised` | supported | supported | supported | supported | not available |
| `nonneg_ridge` | linear | `supervised` | supported | supported | supported | supported | not available |
| `ols` | linear | `supervised` | supported | supported | supported | supported | not available |
| `random_walk_ridge` | linear | `supervised` | supported | supported | supported | supported | not available |
| `ridge` | linear | `supervised` | supported | supported | supported | supported | not available |
| `shrink_to_target_ridge` | linear | `supervised` | supported | supported | supported | supported | not available |
| `sparse_group_lasso` | linear | `supervised` | supported | supported | supported | supported | not available |
| `tvp_ridge` | linear | `supervised` | supported | supported | supported | supported | not available |
| `dfm_mixed_mariano_murasawa` | mixed_frequency | `panel` | guarded-unsupported | guarded-unsupported | supported | supported | not available |
| `dfm_unrestricted_midas` | mixed_frequency | `panel` | guarded-unsupported | guarded-unsupported | supported | supported | not available |
| `midas_almon` | mixed_frequency | `supervised` | supported | supported | supported | supported | not available |
| `midas_beta` | mixed_frequency | `supervised` | supported | supported | supported | supported | not available |
| `midas_step` | mixed_frequency | `supervised` | supported | supported | supported | supported | not available |
| `restricted_midas` | mixed_frequency | `supervised` | supported | supported | supported | supported | not available |
| `unrestricted_midas` | mixed_frequency | `supervised` | supported | supported | supported | supported | not available |
| `density_hnn` | neural | `supervised` | supported | supported | supported | supported | not available |
| `gru` | neural | `supervised` | supported | supported | supported | supported | not available |
| `hemisphere_nn` | neural | `supervised` | supported | supported | supported | supported | not available |
| `lstm` | neural | `supervised` | supported | supported | supported | supported | not available |
| `nn` | neural | `supervised` | supported | supported | supported | supported | not available |
| `transformer` | neural | `supervised` | supported | supported | supported | supported | not available |
| `kernel_ridge` | nonparametric | `supervised` | supported | supported | supported | supported | not available |
| `knn` | nonparametric | `supervised` | supported | supported | supported | supported | not available |
| `mars` | spline | `supervised` | supported | supported | supported | supported | not available |
| `linear_svr` | support_vector | `supervised` | supported | supported | supported | supported | not available |
| `nu_svr` | support_vector | `supervised` | supported | supported | supported | supported | not available |
| `svr` | support_vector | `supervised` | supported | supported | supported | supported | not available |
| `ar` | timeseries | `supervised` | supported-via-direct-projection | supported-via-direct-projection | supported | supported | not available |
| `arima` | timeseries | `target` | guarded-unsupported | guarded-unsupported | supported | supported | not available |
| `auto_arima` | timeseries | `target` | guarded-unsupported | guarded-unsupported | supported | supported | not available |
| `bvar_minnesota` | timeseries | `panel` | guarded-unsupported | guarded-unsupported | supported | supported | not available |
| `bvar_normal_inverse_wishart` | timeseries | `panel` | guarded-unsupported | guarded-unsupported | supported | supported | not available |
| `ets` | timeseries | `target` | guarded-unsupported | guarded-unsupported | supported | supported | not available |
| `holt_winters` | timeseries | `target` | guarded-unsupported | guarded-unsupported | supported | supported | not available |
| `naive` | timeseries | `target` | guarded-unsupported | guarded-unsupported | supported | supported | not available |
| `random_walk_drift` | timeseries | `target` | guarded-unsupported | guarded-unsupported | supported | supported | not available |
| `seasonal_naive` | timeseries | `target` | guarded-unsupported | guarded-unsupported | supported | supported | not available |
| `stlf` | timeseries | `target` | guarded-unsupported | guarded-unsupported | supported | supported | not available |
| `theta_method` | timeseries | `target` | guarded-unsupported | guarded-unsupported | supported | supported | not available |
| `var` | timeseries | `panel` | supported-via-direct-projection | supported-via-direct-projection | supported | supported | not available |
| `catboost` | tree | `supervised` | supported | supported | supported | supported | not available |
| `decision_tree` | tree | `supervised` | supported | supported | supported | supported | not available |
| `extra_trees` | tree | `supervised` | supported | supported | supported | supported | not available |
| `gradient_boosting` | tree | `supervised` | supported | supported | supported | supported | not available |
| `lgb_plus` | tree | `supervised` | supported | supported | supported | supported | not available |
| `lgba_plus` | tree | `supervised` | supported | supported | supported | supported | not available |
| `lightgbm` | tree | `supervised` | supported | supported | supported | supported | not available |
| `macro_random_forest` | tree | `supervised` | supported | supported | supported | supported | not available |
| `quantile_regression_forest` | tree | `supervised` | supported | supported | supported | supported | not available |
| `random_forest` | tree | `supervised` | supported | supported | supported | supported | not available |
| `xgboost` | tree | `supervised` | supported | supported | supported | supported | not available |
| `egarch` | volatility | `volatility` | supported | supported | supported | supported | not available |
| `garch11` | volatility | `volatility` | supported | supported | supported | supported | not available |
| `gjr_garch` | volatility | `volatility` | supported | supported | supported | supported | not available |
| `realized_garch` | volatility | `volatility` | supported | supported | supported | supported | not available |
| `tgarch` | volatility | `volatility` | supported | supported | supported | supported | not available |

Measured scan source unavailable in this checkout (`.dev-notes/policy_matrix_results.json` was not present when this page was generated).
