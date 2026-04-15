# API Reference

## Available package surfaces

The rebuilt macrocast package currently exposes seven documented code surfaces:

- [`macrocast.stage0`](stage0.md)
- [`macrocast.raw`](raw.md)
- [`macrocast.recipes`](recipes.md)
- [`macrocast.preprocessing`](preprocessing.md)
- [`macrocast.registry`](registry.md)
- [`macrocast.compiler`](compiler.md)
- [`macrocast.execution`](execution.md)

## Current operational subset summary

Training
- frameworks: `expanding`, `rolling`
- benchmarks: `historical_mean`, `zero_change`, `ar_bic`
- model families: `ar`, `ridge`, `lasso`, `elasticnet`, `randomforest`
- feature builders: `autoreg_lagged_target`, `raw_feature_panel`

Preprocessing
- explicit `raw_only`
- first train-only path: `extra_preprocess_without_tcode + x_missing_policy=em_impute + scaling_policy=standard`

Evaluation / testing / importance
- metrics: `msfe`, `relative_msfe`, `oos_r2`, `csfe`
- operational stat tests: `dm`, `cw`
- operational importance: `minimal_importance`

Execution architecture
- separate model executor and benchmark executor contracts
- fixed single feature-builder runs operational
- internal feature-builder sweeps not yet operational
