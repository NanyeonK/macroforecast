# Execution pipeline

## Purpose

The execution layer consumes current package contracts and emits deterministic run artifacts.
It sits behind an explicit compiler boundary, preserves preprocessing semantics in output provenance, and treats model execution and benchmark execution as separate runtime components.

## Current role

The current runtime now supports a first importance layer in addition to frameworks, preprocessing, DM testing, and CW testing.
It executes a benchmark-respecting slice with:
- revised-data single-target point forecast
- explicit benchmark family from recipe grammar
- deterministic prediction and metric artifacts
- two operational feature-builder families
- one nontrivial train-only raw-panel preprocessing path
- operational statistical tests: DM and CW
- first operational importance layer: minimal importance

## Current operational frameworks

- `expanding`
- `rolling`

## Current operational statistical tests

- `stat_test = dm`
- `stat_test = cw`
- DM writes `stat_test_dm.json`
- CW writes `stat_test_cw.json`
- CW first slice uses the benchmark-vs-model forecast-gap adjustment on the existing prediction table and reports a simple normal-approximation statistic
- manifest records `stat_test_spec` and `stat_test_file`

## Current operational importance layer

The current runtime can execute:
- `importance_method = minimal_importance`

Current behavior:
- writes `importance_minimal.json`
- manifest records `importance_spec` and `importance_file`
- currently implemented for:
  - non-AR linear route: `ridge`
  - tree route: `randomforest`
- current minimal importance requires `feature_builder='raw_feature_panel'`
- unsupported importance requests fail explicitly at runtime

Current implementation semantics:
- ridge importance = absolute coefficient magnitude from the final fitted training window
- randomforest importance = `feature_importances_` from the final fitted training window

## Current operational feature builders

- `autoreg_lagged_target`
- `raw_feature_panel`

## Current operational preprocessing paths

- explicit `raw_only`
- one train-only raw-panel extra-preprocess path:
  - `tcode_policy = extra_preprocess_without_tcode`
  - `x_missing_policy = em_impute`
  - `scaling_policy = standard`
  - `preprocess_order = extra_only`
  - `preprocess_fit_scope = train_only`

## Current model executors

- `ar`
- `ridge`
- `lasso`
- `elasticnet`
- `randomforest`

## Current benchmark executors

- `historical_mean`
- `zero_change`
- `ar_bic`

## Provenance behavior

The manifest preserves:
- `preprocess_summary`
- full `preprocess_contract`
- `execution_architecture`
- full `model_spec`
- full `benchmark_spec`
- `stat_test_spec`
- `importance_spec`
- optional compiler provenance payload

## Current limitation

Even though minimal importance is now operational, the importance layer is still narrow:
- only `minimal_importance` is operational
- current support is intentionally limited to one linear family and one tree family on the raw-panel path
- SHAP remains future work
