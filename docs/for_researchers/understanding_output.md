# Understanding Output

This page documents the current `macroforecast.core.runtime` output shape. Older `macroforecast.execution` runs may still write legacy files such as `predictions.csv`, `metrics.json`, and `artifact_manifest.json`; the layer-contract runtime writes the L8 directory described here.

## L8 Directory Structure

With `8_output` enabled, `execute_minimal_forecast` writes to `leaf_config.output_directory`.

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

L5 point and relative metrics. Current core runtime materializes:

| Metric | Definition |
|---|---|
| `mse` | Mean squared error |
| `rmse` | Square root of MSE |
| `mae` | Mean absolute error |
| `relative_mse` | Model MSE divided by benchmark MSE, when exactly one benchmark exists |
| `r2_oos` | `1 - relative_mse` |
| `relative_mae` | Model MAE divided by benchmark MAE |
| `mse_reduction` | Benchmark MSE minus model MSE |

## ranking.csv

The L5 ranking table sorted by the resolved ranking metric. For loss metrics, lower is better. For `r2_oos` and `mse_reduction`, higher is better.

## tests_summary.json

L6 writes a JSON representation of `L6TestsArtifact` when `saved_objects` includes `tests`.

Current core runtime can populate:

- `equal_predictive_results`
- `nested_results`
- `cpa_results`
- `multiple_model_results`
- `direction_results`
- `residual_results`

The values are descriptive/minimal test dictionaries. Exact bootstrap MCS/SPA/RC/StepM, HAC critical values, and density/interval tests require specialized runtime support.

## importance_summary.json

L7 writes a JSON representation of `L7ImportanceArtifact` when `saved_objects` includes `importance`.

Current core runtime can populate:

- `global_importance` from linear coefficients and permutation-style importance,
- `group_importance` from `group_aggregate`,
- `lineage_importance` from L3 metadata,
- `L7TransformationAttributionArtifact` when `l7_transformation_attribution_v1` is requested.

Advanced SHAP backends, neural-gradient methods, VAR FEVD/IRF, and figure rendering are outside the current core runtime path.

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
