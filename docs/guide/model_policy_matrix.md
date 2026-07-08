# Model x Forecast Policy Matrix

[Back to Models and Features](model_overview.md)

This page states which forecast policies are statistically supported for each
registered model. It is generated from `mf.list_model_specs()`, each model's
`ModelSpec.default_params`, and the package's direct-policy guard.

Status meanings:

- `supported`: the runner policy applies to this model family.
- `supported-via-direct-projection`: the model has an explicit validated direct
  projection mode for this policy.
- `guarded-unsupported`: the combination is blocked by default because the model
  forecasts by iterating its own dynamics rather than by fitting an h-step direct
  projection, or because the model's direct projection is a point target rather
  than the requested horizon-average target.

The measured-scan column is only a point-in-time runtime smoke scan. `OK` means a
combination ran, not that its forecast object has the right statistical meaning.

| Model | Family | Input | direct | direct_average | path_average | recursive | Measured scan |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `albacore_components` | assemblage | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `albacore_ranks` | assemblage | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `assemblage_regression` | assemblage | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `component_aggregation` | assemblage | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `rank_aggregation` | assemblage | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `supervised_aggregation` | assemblage | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `ucsv` | bayesian | `target` | guarded-unsupported | guarded-unsupported | supported | supported | not available |
| `pls` | composite | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `scaled_pca` | composite | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `supervised_pca` | composite | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `supervised_scaled_pca` | composite | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `far` | factor | `supervised` | supported-via-direct-projection | supported-via-direct-projection | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `favar` | factor | `supervised` | guarded-unsupported | guarded-unsupported | supported | supported | direct: EMPTY, direct_average: EMPTY, path_average: EMPTY, recursive: EMPTY |
| `adaptive_elastic_net` | linear | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `adaptive_lasso` | linear | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `bayesian_ridge` | linear | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `elastic_net` | linear | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `fused_difference_ridge` | linear | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `glmboost` | linear | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `group_lasso` | linear | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `huber` | linear | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `lasso` | linear | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `nonneg_ridge` | linear | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `ols` | linear | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `random_walk_ridge` | linear | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `ridge` | linear | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `shrink_to_target_ridge` | linear | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `sparse_group_lasso` | linear | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `tvp_ridge` | linear | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `dfm_mixed_mariano_murasawa` | mixed_frequency | `panel` | guarded-unsupported | guarded-unsupported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: EMPTY |
| `dfm_unrestricted_midas` | mixed_frequency | `panel` | guarded-unsupported | guarded-unsupported | supported | supported | direct: EMPTY, direct_average: EMPTY, path_average: EMPTY, recursive: EMPTY |
| `midas_almon` | mixed_frequency | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `midas_beta` | mixed_frequency | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `midas_step` | mixed_frequency | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `restricted_midas` | mixed_frequency | `supervised` | supported | supported | supported | supported | direct: TIMEOUT, direct_average: TIMEOUT, path_average: TIMEOUT, recursive: TIMEOUT |
| `unrestricted_midas` | mixed_frequency | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `csr` | model_averaging | `supervised` | supported | supported | supported | supported | not available |
| `jma` | model_averaging | `supervised` | supported | supported | supported | supported | not available |
| `density_hnn` | neural | `supervised` | supported | supported | supported | supported | direct: EMPTY, direct_average: EMPTY, path_average: EMPTY, recursive: EMPTY |
| `gru` | neural | `supervised` | supported | supported | supported | supported | direct: EMPTY, direct_average: EMPTY, path_average: EMPTY, recursive: EMPTY |
| `hemisphere_nn` | neural | `supervised` | supported | supported | supported | supported | direct: EMPTY, direct_average: EMPTY, path_average: EMPTY, recursive: EMPTY |
| `lstm` | neural | `supervised` | supported | supported | supported | supported | direct: EMPTY, direct_average: EMPTY, path_average: EMPTY, recursive: EMPTY |
| `nn` | neural | `supervised` | supported | supported | supported | supported | direct: EMPTY, direct_average: EMPTY, path_average: EMPTY, recursive: EMPTY |
| `transformer` | neural | `supervised` | supported | supported | supported | supported | direct: EMPTY, direct_average: EMPTY, path_average: EMPTY, recursive: EMPTY |
| `kernel_ridge` | nonparametric | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `knn` | nonparametric | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `mars` | spline | `supervised` | supported | supported | supported | supported | direct: TIMEOUT, direct_average: TIMEOUT, path_average: TIMEOUT, recursive: TIMEOUT |
| `linear_svr` | support_vector | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `nu_svr` | support_vector | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `svr` | support_vector | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `ar` | timeseries | `supervised` | supported-via-direct-projection | supported-via-direct-projection | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `arima` | timeseries | `target` | guarded-unsupported | guarded-unsupported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `auto_arima` | timeseries | `target` | guarded-unsupported | guarded-unsupported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `bvar_minnesota` | timeseries | `panel` | guarded-unsupported | guarded-unsupported | supported | supported | direct: TIMEOUT, direct_average: TIMEOUT, path_average: TIMEOUT, recursive: TIMEOUT |
| `bvar_normal_inverse_wishart` | timeseries | `panel` | guarded-unsupported | guarded-unsupported | supported | supported | direct: TIMEOUT, direct_average: TIMEOUT, path_average: TIMEOUT, recursive: TIMEOUT |
| `ets` | timeseries | `target` | guarded-unsupported | guarded-unsupported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `hist_mean` | timeseries | `target` | supported | supported | supported | supported | not available |
| `holt_winters` | timeseries | `target` | guarded-unsupported | guarded-unsupported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `naive` | timeseries | `target` | guarded-unsupported | guarded-unsupported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `random_walk_drift` | timeseries | `target` | guarded-unsupported | guarded-unsupported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `seasonal_naive` | timeseries | `target` | guarded-unsupported | guarded-unsupported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `stlf` | timeseries | `target` | guarded-unsupported | guarded-unsupported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `theta_method` | timeseries | `target` | guarded-unsupported | guarded-unsupported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `var` | timeseries | `panel` | supported-via-direct-projection | guarded-unsupported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: EMPTY |
| `catboost` | tree | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `decision_tree` | tree | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `extra_trees` | tree | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `gradient_boosting` | tree | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `lgb_plus` | tree | `supervised` | supported | supported | supported | supported | direct: TIMEOUT, direct_average: TIMEOUT, path_average: TIMEOUT, recursive: TIMEOUT |
| `lgba_plus` | tree | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `lightgbm` | tree | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `macro_random_forest` | tree | `supervised` | supported | supported | supported | supported | direct: TIMEOUT, direct_average: TIMEOUT, path_average: TIMEOUT, recursive: TIMEOUT |
| `quantile_regression_forest` | tree | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `random_forest` | tree | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `xgboost` | tree | `supervised` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `egarch` | volatility | `volatility` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `garch11` | volatility | `volatility` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `gjr_garch` | volatility | `volatility` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `realized_garch` | volatility | `volatility` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |
| `tgarch` | volatility | `volatility` | supported | supported | supported | supported | direct: OK, direct_average: OK, path_average: OK, recursive: OK |

Measured scan source: `.dev-notes/policy_matrix_results.json`. Treat it as a runtime smoke scan, not as support metadata.
