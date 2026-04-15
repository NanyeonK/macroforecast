# Compiler guide

## Purpose

The compiler is the bridge between:
- recipe YAML
- canonical grammar
- runtime execution eligibility

It does not silently coerce unsupported choices into whatever currently runs.

## Current framework executability rule

Executable frameworks:
- `expanding`
- `rolling`

## Current statistical-test executability rule

Executable stat tests:
- `none`
- `dm`
- `cw`

## Current importance executability rule

Executable importance methods:
- `none`
- `minimal_importance`

Current minimal-importance rule:
- compiler preserves the request as `importance_spec`
- runtime currently supports the first operational slice on compatible fixed single-run routes
- unsupported importance requests still fail explicitly at runtime rather than silently degrading

## Current executable slice

The current compiler-to-runtime path supports:
- frameworks: `expanding`, `rolling`
- benchmarks: `historical_mean`, `zero_change`, `ar_bic`, `custom_benchmark`
- model families: `ar`, `ridge`, `lasso`, `elasticnet`, `randomforest`
- feature builders: `autoreg_lagged_target`, `raw_feature_panel`
- preprocessing: `raw_only` plus the first train-only impute+standardize raw-panel path
- always-written comparison artifact: `comparison_summary.json`
- statistical tests: `dm`, `cw`
- importance: `minimal_importance`

Important caveat
- fixed single feature-builder runs are executable
- internal multi-value feature-builder sweeps are still representable but not executable in one compiled run
- incompatible requests such as `model_family='ar'` with `feature_builder='raw_feature_panel'` are explicitly blocked
- `custom_benchmark` is executable only through the first plugin-ready bridge and currently requires `benchmark_config.plugin_path` plus `benchmark_config.callable_name`
