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
- current supported model families are `ridge`, `lasso`, and `randomforest` on `raw_feature_panel`
- unsupported importance requests still fail explicitly at runtime rather than silently degrading

## Current executable slice

Stage 1 note: compiler now accepts canonical data/task axes like `information_set_type`, `data_domain`, `dataset_source`, `predictor_family`, and related provenance axes. Legacy `info_set` remains a backward-compat alias to `information_set_type`.


The current compiler-to-runtime path supports:
- frameworks: `expanding`, `rolling`
- benchmarks: `historical_mean`, `zero_change`, `ar_bic`, `custom_benchmark`
- model families: `ar`, `ridge`, `lasso`, `elasticnet`, `randomforest`
- feature builders: `autoreg_lagged_target`, `raw_feature_panel`
- info sets: `revised` and the first `real_time` explicit-vintage slice
- tasks: `single_target_point_forecast` plus the first narrow `multi_target_point_forecast` slice
- preprocessing: `raw_only` plus train-only impute+standardize and train-only impute+robust-scale raw-panel paths
- always-written comparison artifact: `comparison_summary.json`
- statistical tests: `dm`, `cw`
- importance: `minimal_importance`

Important caveat
- fixed single feature-builder runs are executable
- the current multi-target slice requires explicit `leaf_config.targets` and keeps one shared model/benchmark/preprocess environment across all targets
- wrapper-owned studies are not executable runs; compiler emits `wrapper_handoff` instead and requires `leaf_config.wrapper_family` plus `leaf_config.bundle_label`
- compiler now also emits deterministic `tree_context` provenance so compile artifacts preserve Stage 0 fixed design, varying design, and path-level fixed/sweep/conditional grouping
- internal multi-value feature-builder sweeps are still representable but not executable in one compiled run
- incompatible requests such as `model_family='ar'` with `feature_builder='raw_feature_panel'` are explicitly blocked
- `custom_benchmark` is executable only through the first plugin-ready bridge and currently requires `benchmark_config.plugin_path` plus `benchmark_config.callable_name`
- the current `real_time` slice is explicit-vintage only and requires `leaf_config.data_vintage`; it is not yet a rolling historical real-time panel engine
