# Compatibility Engine

The Compatibility Engine is the constraint-aware view over the registry and compiler. It does not merely list options. It explains which choices remain valid after the current selection.

The CLI and static Navigator App share the same exported compatibility metadata. In UI data this lives under `state_engine` with schema version `navigator_state_engine_v1`. The browser uses that payload to recompute option status, disabled reasons, compatibility messages, and YAML preview output while the researcher edits a path.

## Current Rule Families

| Current selection | Effect |
|---|---|
| `importance_method=tree_shap` | Keeps tree generators: `random_forest`, `extra_trees`, `gradient_boosting`, `xgboost`, `lightgbm`, `catboost`. Non-tree models are disabled. |
| `importance_method=linear_shap` | Keeps linear estimators such as `ridge`, `lasso`, `elasticnet`, `bayesian_ridge`, `huber`, `adaptive_lasso`, and `quantile_linear`. |
| `importance_shap=tree_shap` | Same tree-model restriction as the legacy single `importance_method` route. |
| `importance_shap=linear_shap` | Same linear-estimator restriction as the legacy single `importance_method` route. |
| `importance_model_native=minimal_importance` | Requires the current raw-panel importance runtime. |
| Layer 7 detail axes | `importance_scope`, `importance_aggregation`, `importance_output_style`, `importance_temporal`, and `importance_gradient_path` stay on operational defaults unless an importance family is active. |
| `forecast_object=quantile` | Keeps `model_family=quantile_linear` as the operational generator. Downstream quantile metrics/tests should be preferred where available. |
| `forecast_object=direction` | Recommends direction tests such as `pesaran_timmermann` and `binomial_hit`. |
| `forecast_object=interval` or `density` | Recommends density/interval calibration tests on the `density_interval` axis. |
| `model_family in {lstm, gru, tcn}` | Current runtime uses the univariate target-history sequence route. Full multivariate `feature_runtime=sequence_tensor` remains gated until the Layer 2 sequence representation handoff is opened. |
| registered custom `model_family` | Enabled in the current Python process after `@mf.custom_model(...)` registration. Custom names are valid forecast generators, but YAML alone cannot register the callable. |
| `fred_sd_mixed_frequency_representation=native_frequency_block_payload` | Requires `dataset` including `fred_sd`, `feature_builder=raw_feature_panel`, and `forecast_type=direct`. It enables FRED-SD native-frequency payloads for registered custom models and supported MIDAS routes. |
| `fred_sd_mixed_frequency_representation=mixed_frequency_model_adapter` | Requires the same FRED-SD raw-panel direct route and enables the adapter payload for registered custom models, `midas_almon`, `midasr`, and `midasr_nealmon`. |
| `forecast_type=iterated` with raw-panel features | Requires an explicit `exogenous_x_path_policy`: `hold_last_observed`, `observed_future_x`, `scheduled_known_future_x`, or `recursive_x_model` with `recursive_x_model_family=ar1`. |
| `export_format=parquet` or `all` | Adds sidecar artifact files; the CSV prediction table remains the stable baseline artifact. |
| `saved_objects=predictions_only` | Saves manifest, run summary, predictions, and forecast payload files only; Layer 4 metrics/reports and Layer 6/7 artifacts are not materialized. |
| `saved_objects=predictions_and_metrics` | Adds Layer 4 metrics, comparison, and evaluation-summary artifacts but skips full-bundle data/model/tuning/inference/importance sidecars. |
| `artifact_granularity != aggregated` | Disabled in the current runtime because per-target/per-horizon/hierarchical result-object readers are not implemented. |
| `regime_definition != none` | Treats regime handling as post-forecast evaluation filtering unless a future training-time regime gate is opened. |
| `direction` statistical tests | Enabled for `forecast_object=direction`; otherwise direction tests stay inactive. |
| HAC/bootstrap dependence correction | Attached to HAC/bootstrap-compatible Layer 6 choices across legacy `stat_test` and split axes. `evaluate_with_hac` stays disabled when any active test is not HAC-capable. |

## Why Disabled Branches Matter

Disabled branches are not documentation warnings. They are part of the route contract.

Examples:

```text
importance_method=tree_shap
model_family=ridge -> disabled: tree_shap requires a tree model
```

```text
importance_shap=tree_shap
model_family=ridge -> disabled: tree_shap requires a tree model
```

```text
forecast_object=quantile
model_family=ridge -> disabled: quantile currently requires model_family=quantile_linear
```

```text
feature_builder=raw_feature_panel
forecast_type=iterated
exogenous_x_path_policy=unavailable -> disabled for executable raw-panel iterated runs
```

```text
fred_sd_mixed_frequency_representation=mixed_frequency_model_adapter
feature_builder=target_lag_features -> disabled: FRED-SD adapter payloads require raw_feature_panel
```

```text
model_family=my_custom_model in YAML
callable not registered in Python process -> disabled at compile/run: custom model name is unknown
```

## Operational Versus Named-Gated

The docs distinguish values that exist in the grammar from values that are executable.

| Status | Interpretation |
|---|---|
| `operational` | Current runtime can execute the value. |
| `operational_narrow` | Current runtime executes a named narrow slice. |
| `registry_only` | Value is named but not wired to an executable runtime. |
| `future` | Future design value. |
| `external_plugin` | Requires a registered external callable/plugin. |
| `gated_named` | Contract has a name, but the runtime gate is closed. |

## Optional Backends

Optional backends are not loaded during navigation:

- `optuna` is loaded only for bayesian tuning;
- `xgboost`, `lightgbm`, and `catboost` are loaded only when their model family is selected.

This keeps the documentation/navigation path light while preserving the larger method surface.
