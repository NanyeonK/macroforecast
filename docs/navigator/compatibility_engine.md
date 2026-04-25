# Compatibility Engine

The Compatibility Engine is the constraint-aware view over the registry and compiler. It does not merely list options. It explains which choices remain valid after the current selection.

## Current Rule Families

| Current selection | Effect |
|---|---|
| `importance_method=tree_shap` | Keeps tree generators: `randomforest`, `extratrees`, `gbm`, `xgboost`, `lightgbm`, `catboost`. Non-tree models are disabled. |
| `importance_method=linear_shap` | Keeps linear estimators such as `ridge`, `lasso`, `elasticnet`, `bayesianridge`, `huber`, `adaptivelasso`, and `quantile_linear`. |
| `forecast_object=quantile` | Keeps `model_family=quantile_linear` as the operational generator. Downstream quantile metrics/tests should be preferred where available. |
| `forecast_object=direction` | Recommends direction tests such as `pesaran_timmermann` and `binomial_hit`. |
| `forecast_object=interval` or `density` | Recommends density/interval calibration tests on the `density_interval` axis. |
| `model_family in {lstm, gru, tcn}` | Current runtime uses the univariate target-history sequence route. Full multivariate `feature_runtime=sequence_tensor` remains gated until the Layer 2 sequence representation handoff is opened. |
| `forecast_type=iterated` with raw-panel features | Requires an explicit `exogenous_x_path_policy`: `hold_last_observed`, `observed_future_x`, `scheduled_known_future_x`, or `recursive_x_model` with `recursive_x_model_family=ar1`. |
| `export_format=parquet` or `all` | Adds sidecar artifact files; the CSV prediction table remains the stable baseline artifact. |
| `saved_objects=predictions_only` | Saves manifest, run summary, predictions, and forecast payload files only; Layer 4 metrics/reports and Layer 6/7 artifacts are not materialized. |
| `saved_objects=predictions_and_metrics` | Adds Layer 4 metrics, comparison, and evaluation-summary artifacts but skips full-bundle data/model/tuning/inference/importance sidecars. |
| `artifact_granularity != aggregated` | Disabled in the current runtime because per-target/per-horizon/hierarchical result-object readers are not implemented. |
| `regime_definition != none` | Treats regime handling as post-forecast evaluation filtering unless a future training-time regime gate is opened. |
| `direction` statistical tests | Enabled for `forecast_object=direction`; otherwise direction tests stay inactive. |
| HAC/bootstrap dependence correction | Attached to HAC/bootstrap-compatible test choices. Split Layer 6 axes are visible now; runtime harmonization is handled in the Layer 6 cleanup slice. |

## Why Disabled Branches Matter

Disabled branches are not documentation warnings. They are part of the route contract.

Examples:

```text
importance_method=tree_shap
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
