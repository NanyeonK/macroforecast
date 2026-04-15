# API Reference

## Available package surfaces

The rebuilt macrocast package currently exposes eight documented code surfaces:

- [`macrocast.stage0`](stage0.md)
- [`macrocast.raw`](raw.md)
- [`macrocast.recipes`](recipes.md)
- [`macrocast.preprocessing`](preprocessing.md)
- [`macrocast.registry`](registry.md)
- [`macrocast.compiler`](compiler.md)
- [`macrocast.execution`](execution.md)
- [`macrocast.start`](start.md)

## Current operational subset summary

Training
- frameworks: `expanding`, `rolling`
- benchmarks: `historical_mean`, `zero_change`, `ar_bic`, `custom_benchmark`
- model families: `ar`, `ridge`, `lasso`, `elasticnet`, `randomforest`
- feature builders: `autoreg_lagged_target`, `raw_feature_panel`

Data / task
- info sets: `revised`, `real_time` (current real-time slice requires explicit `data_vintage`)
- tasks currently operational:
  - `single_target_point_forecast`
  - first narrow `multi_target_point_forecast` slice with explicit `targets`

Preprocessing
- explicit `raw_only`
- train-only raw-panel path: `extra_preprocess_without_tcode + x_missing_policy=em_impute + scaling_policy=standard`
- train-only raw-panel path: `extra_preprocess_without_tcode + x_missing_policy=em_impute + scaling_policy=robust`

Evaluation / testing / importance
- metrics: `msfe`, `relative_msfe`, `oos_r2`, `csfe`
- always-written comparison artifact: `comparison_summary.json`
- operational stat tests: `dm`, `cw`
- operational importance: `minimal_importance` for `ridge`, `lasso`, and `randomforest` on `raw_feature_panel`

Execution architecture
- separate model executor and benchmark executor contracts
- fixed single feature-builder runs operational
- internal feature-builder sweeps not yet operational


Single-run route inspection
- `macrocast_single_run(yaml_path=...)` now exposes route preview, compile preview, tree-context preview, and honest blocking of run/manifest previews for non-executable or wrapper-owned routes.

- omitting `yaml_path` now starts a minimal staged selector that rewrites YAML step-by-step and refreshes route preview after each completed choice.

- the staged selector now covers framework / benchmark / narrow operational preprocessing choices before model-path choices.

- the staged selector now also covers evaluation / output / stat-test / importance choices in the current executable single-run subset.

- the staged selector now labels planned options from the live registry and surfaces explicit planned-branch messages when those options are chosen.

- the staged selector now exposes `model_path_mode` and explicitly distinguishes model-grid vs full-sweep planned single-run extensions.
