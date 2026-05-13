# Understanding Output

This page documents the v0.9.0 L8 output directory written by `mf.run(...)` / `mf.forecast(...)` / `mf.Experiment(...)`. There is no legacy file layout to disambiguate against.

## L8 Directory Structure

With `8_output` enabled, the runtime writes to `leaf_config.output_directory`.

Typical directory:

```text
macroforecast_output/<name>/
  manifest.json              # L8 provenance and saved object inventory
  recipe.json                # JSON copy of the resolved input recipe
  summary/
    metrics_all_cells.csv    # L5 metrics table
    ranking.csv              # L5 ranking table
  cell_001/
    forecasts.csv            # L4 point forecasts
    clean_panel.csv          # optional, when saved_objects includes clean_panel
    raw_panel.csv            # optional, when saved_objects includes raw_panel
    feature_metadata.json    # optional, when saved_objects includes feature_metadata
  diagnostics/
    l1_5_diagnostic_v1.json  # optional
    l2_5_diagnostic_v1.json  # optional
    l3_5_diagnostic_v1.json  # optional
    l4_5_diagnostic_v1.json  # optional
  tests_summary.json         # optional, when L6 is saved
  importance_summary.json    # optional, when L7 is saved
```

`L8ArtifactsArtifact.exported_files` records the files written during the run.

## forecasts.csv

One row per `(model_id, target, horizon, origin)` forecast.

| Column | Meaning |
|---|---|
| `model_id` | L4 fit node id |
| `target` | Forecast target |
| `horizon` | Forecast horizon |
| `origin` | Forecast origin date |
| `forecast` | Point forecast |

## metrics_all_cells.csv

L5 metrics organised into four families.

**Point metrics**

| Metric | Definition |
|---|---|
| `mse` | Mean squared error |
| `rmse` | Square root of MSE |
| `mae` | Mean absolute error |
| `med_ae` | Median absolute error |
| `mape` | Mean absolute percentage error |
| `theil_u1` | Theil U1 |
| `theil_u2` | Theil U2 |

**Relative metrics** (require exactly one benchmark)

| Metric | Definition |
|---|---|
| `relative_mse` | Model MSE divided by benchmark MSE |
| `relative_mae` | Model MAE divided by benchmark MAE |
| `mse_reduction` | Benchmark MSE minus model MSE |
| `r2_oos` | `1 - relative_mse` |

**Density metrics**

| Metric | Definition |
|---|---|
| `log_score` | Mean log predictive density |
| `crps` | Continuous Ranked Probability Score |
| `interval_score` | Winkler interval score |
| `coverage_rate` | Empirical coverage of prediction intervals |

**Direction metrics**

| Metric | Definition |
|---|---|
| `success_ratio` | Sign-match fraction |
| `pesaran_timmermann` | Pesaran-Timmermann directional accuracy statistic |

**Aggregations** controlled by L5.B/C/D axes: `per_subperiod`, `by_predictor_block` (Shapley-share), `per_horizon_then_mean`, `top_k_worst`.

## ranking.csv

The L5 ranking table sorted by the resolved ranking metric. For loss metrics, lower is better. For `r2_oos` and `mse_reduction`, higher is better.

## tests_summary.json

L6 writes a JSON representation of `L6TestsArtifact` when `saved_objects` includes `tests`.

The runtime populates:

- `equal_predictive_results` -- DM (HLN), Giacomini-White, Diebold-Mariano-Pesaran multi-horizon, HLN encompassing; Newey-West / Andrews / Parzen HAC kernels
- `nested_results` -- Clark-West
- `cpa_results` -- Giacomini-Rossi 2010 rolling fluctuation (simulated CV) + Rossi-Sekhposyan
- `multiple_model_results` -- Hansen MCS, SPA, White Reality Check, Romano-Wolf StepM (stationary bootstrap, auto-tuned block length)
- `density_results` -- Engle-Manganelli DQ, log_score / CRPS pair tests
- `direction_results` -- Pesaran-Timmermann
- `residual_results` -- autocorrelation, heteroskedasticity, normality diagnostics

## importance_summary.json

L7 writes a JSON representation of `L7ImportanceArtifact` when `saved_objects` includes `importance`.

The runtime populates:

- `global_importance` -- linear coef, MDI tree, BFR permutation, Strobl unbiased permutation, LOFO
- `shap_importance` -- SHAP tree / kernel / linear / interaction / deep (`[shap]` extra)
- `gradient_importance` -- gradient_shap, integrated_gradients, saliency_map, deep_lift via captum (`[deep]` extra)
- `effect_importance` -- partial_dependence, ALE, Friedman H^2
- `var_decomposition` -- FEVD, historical_decomposition, orthogonalised_irf, generalized_irf
- `group_importance` from `group_aggregate`; `lineage_importance` from `lineage_attribution`; `L7TransformationAttributionArtifact` from `transformation_attribution`
- `temporal_importance` -- `rolling_recompute`, `bootstrap_jackknife`
- Advanced: `mrf_gtvp`, `dual_decomposition`, `oshapley_vi`, `pbsv`, `attention_weights`

Figure rendering for these artifacts is governed by the L7.B `figure_type` / `figure_format` axes.

## Diagnostic JSON Files

Diagnostic layers are non-blocking. They produce `DiagnosticArtifact` JSON payloads when enabled:

| Sink | Contents |
|---|---|
| `l1_5_diagnostic_v1` | sample coverage, univariate summaries, missing/outlier audit, optional correlation |
| `l2_5_diagnostic_v1` | raw-vs-clean comparison, distribution shift, cleaning effect summary |
| `l3_5_diagnostic_v1` | raw/clean/features comparison, feature summary, lineage summary, lag/factor/selection metadata |
| `l4_5_diagnostic_v1` | forecast/model/training summaries, fit summary, window summary |

The current core runtime writes JSON metadata only. PNG/PDF/HTML/LaTeX rendering is a future specialized diagnostics renderer.

## manifest.json

The L8 manifest records:

- runtime environment summary,
- recipe hash,
- saved object list,
- upstream sink names,
- exported file list,
- dependency lockfile paths when detected.

It is designed to make the output directory inspectable without rerunning the forecast.

## In-Memory Sinks

The filesystem output is a projection of the in-memory runtime result:

```python
result.sink("l4_forecasts_v1")
result.sink("l5_evaluation_v1")
result.sink("l6_tests_v1")
result.sink("l7_importance_v1")
result.sink("l8_artifacts_v1")
```

Use in-memory sinks for tests and notebooks. Use L8 output for handoff, audit, and replication folders.
