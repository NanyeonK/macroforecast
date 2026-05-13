# Runtime Support Matrix

This page describes what the v0.9.0 core runtime executes directly.

macroforecast's runtime surface is `macroforecast.core.runtime`. The public entry point is `macroforecast.run("recipe.yaml")`; the support matrix below refers to the layer-contract runtime that backs it.

## End-to-End Supported Path

The core runtime executes one complete layer-contract path:

1. L1 loads a custom inline/CSV/Parquet panel, or official FRED-MD/FRED-QD/FRED-SD from the raw adapters.
2. L2 applies selected transform, outlier, imputation, and frame-edge policies.
3. L3 executes the feature DAG (deterministic ops plus McCracken-Ng PCA cascades).
4. L4 fits expanding/rolling-window forecasters across 35+ families and produces point/quantile/density forecasts.
5. L5 computes point, relative, density, and direction metrics across multiple aggregations.
6. L1.5-L4.5 materialize diagnostic JSON artifacts when requested.
7. L6 materializes statistical-test result dictionaries from realized forecast errors.
8. L7 materializes importance artifacts (linear/tree/permutation/SHAP/gradient/effect/VAR/temporal families) from fitted models and L3 metadata.
9. L8 writes a reproducible output directory with manifest, recipe, CSV, and JSON sidecars.

## Layer Support

| Layer | Runtime status | Supported now | Not yet full runtime |
|---|---|---|---|
| L0 Meta | Runtime | Study scope, failure_policy, reproducibility_mode, compute_mode, seed pass-through; manifest metadata | -- |
| L1 Data | Runtime | Custom panel inline/records/CSV/Parquet; official FRED-MD/FRED-QD loaders; FRED-SD with state-group / variable-group / mixed-frequency routes; missing_availability / raw_missing_policy / raw_outlier_policy / release_lag_rule / contemporaneous_x_rule | -- |
| L1.5 Data summary | Runtime artifact | Sample coverage, summary stats, ADF/PP/KPSS, missing/outlier audit, optional correlation as JSON | Figure/table rendering |
| L2 Preprocessing | Runtime | Transform (no_transform / t-code), outlier (IQR / z-score / winsorize / MAD), imputation (mean / ffill / interpolation / EM-factor), frame-edge; mixed-frequency alignment | -- |
| L2.5 Pre/post diagnostics | Runtime artifact | Raw-vs-clean comparison, distribution shift, cleaning effect summary, optional correlation shift | Figure/table rendering |
| L3 Features | Runtime | Source, lag, seasonal lag, MA, MARX, concat, scale, log/diff/log_diff/pct_change, polynomial, interaction, season dummy, trend, target_construction, McCracken-Ng PCA, cascade pipelines | -- |
| L3.5 Feature diagnostics | Runtime artifact | Raw/clean/features comparison, feature summary, lineage summary, correlation, lag/factor/selection flags | Factor scree / loadings / selection-stability figure rendering |
| L4 Forecasting | Runtime | Expanding/rolling fit for 35+ families: linear (ols, ridge, lasso, elastic_net, bayesian_ridge, huber, ar_p, factor_augmented_ar), tree (random_forest, extra_trees, gradient_boosting, decision_tree, xgboost, lightgbm, catboost, bagging, quantile_regression_forest, mrf), kernel/NN (svr, mlp, lstm, gru, transformer via `[deep]`), VAR family (var, bvar_minnesota, bvar_normal_inverse_wishart, factor_augmented_var, dfm_mixed_mariano_murasawa), GARCH via `[arch]`; point/quantile/density forecasts; benchmark flag propagation | -- |
| L4.5 Generator diagnostics | Runtime artifact | Forecast/model/training/fit/window summaries; tuning traces; ensemble weight tables | Residual figure rendering |
| L5 Evaluation | Runtime | Point (MSE, RMSE, MAE, MedAE, MAPE, Theil U1/U2), relative (relative_mse, relative_mae, mse_reduction, r2_oos), density (log_score, CRPS, interval_score, coverage_rate), direction (success_ratio, Pesaran-Timmermann); per_subperiod, by_predictor_block (Shapley-share), per_horizon_then_mean, top_k_worst aggregations | -- |
| L6 Statistical tests | Runtime | DM (HLN), Giacomini-White, Diebold-Mariano-Pesaran multi-horizon, HLN encompassing; Clark-West nested; Giacomini-Rossi 2010 (simulated CV) + Rossi-Sekhposyan; Hansen MCS, SPA, White Reality Check, Romano-Wolf StepM; Newey-West / Andrews / Parzen HAC; Engle-Manganelli DQ density | -- |
| L7 Interpretation | Runtime | 30 importance ops: model-native (linear_coef, tree MDI), permutation (BFR/Strobl, LOFO), SHAP (tree/kernel/linear/interaction/deep), gradient (gradient_shap, integrated_gradients, saliency, deep_lift via `[deep]`+captum), effect (PDP, ALE, Friedman H^2), stability (lasso_inclusion_frequency, BVAR PIP, bootstrap_jackknife, rolling_recompute), VAR shock-decomposition (FEVD, historical_decomposition, orthogonalised_irf, generalized_irf), forecast_decomposition, group_aggregate, lineage_attribution, transformation_attribution, MRF GTVP, dual_decomposition, OShapley_VI, PBSV, attention_weights | -- |
| L8 Output | Runtime | Output dir, manifest JSON/JSONL, recipe JSON, forecasts CSV, metrics_all_cells / ranking CSV, tests / importance / diagnostic JSON; LaTeX + Markdown table export | Parquet compression; full HTML report rendering |

## Practical Meaning

Use the core runtime today when you need a reproducible, inspectable forecasting study with layer artifacts. It is appropriate for:

- smoke tests for a full recipe shape,
- custom-panel regression checks,
- benchmark-relative point/quantile/density forecast evaluation,
- statistical-test horse races (Hansen MCS / SPA / Reality Check / Romano-Wolf StepM and friends),
- model interpretation (linear / tree / permutation / SHAP / gradient / effect / VAR / temporal),
- output-directory/provenance integration tests.

The core runtime is the single execution surface in v0.9.0. Everything in the table above is reachable through `mf.run("recipe.yaml")` or the equivalent `mf.forecast(...)` / `mf.Experiment(...)` simple-API entry points.

## Minimal Core Runtime Example

```python
import macroforecast as mf

result = mf.run("examples/recipes/l4_minimal_ridge.yaml")

print(result.sink("l5_evaluation_v1").metrics_table)
```

If `8_output` is included in the recipe, inspect `result.sink("l8_artifacts_v1").output_directory` for exported files.
