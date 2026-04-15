# macrocast

> Given a standardized macro dataset adapter and a fixed forecasting recipe, compare forecasting tools under identical information set, sample split, benchmark, and evaluation protocol.

macrocast is being rebuilt as an architecture-first forecasting package.
The package goal is not to make every long-run choice executable immediately.
The goal is to make the full research choice space explicit in package grammar while systematically promoting registry-defined choices into operational support.

## Rebuild status

The rebuilt package currently has seven public layers/surfaces wired in order:
- `macrocast.stage0`
- `macrocast.raw`
- `macrocast.recipes`
- `macrocast.preprocessing`
- `macrocast.registry`
- `macrocast.compiler`
- `macrocast.execution`

Current operational subset
- operational frameworks: `expanding`, `rolling`
- operational benchmark families: `historical_mean`, `zero_change`, `ar_bic`, `custom_benchmark`
- operational model families: `ar`, `ridge`, `lasso`, `elasticnet`, `randomforest`
- operational feature builders: `autoreg_lagged_target`, `raw_feature_panel`
- operational preprocessing paths:
  - explicit `raw_only`
  - train-only raw-panel path with `x_missing_policy=em_impute` and `scaling_policy=standard`
  - train-only raw-panel path with `x_missing_policy=em_impute` and `scaling_policy=robust`
- operational statistical tests:
  - `dm`
  - `cw`
- operational importance methods:
  - `minimal_importance`
  - current supported routes: `ridge`, `lasso`, `randomforest` on `raw_feature_panel`

Current roadmap focus
- post-wrapper provenance slice now records deterministic `tree_context` payloads in compile/run artifacts so fixed-vs-sweep semantics remain explicit.
- route-inspection preview slice now exists through `macrocast_single_run(yaml_path=...)`, exposing route owner, compile status, and tree-context preview without hidden execution.
- a first minimal staged selector now also exists when `yaml_path` is omitted; it rewrites YAML incrementally for route-defining choices and refreshes route preview after each step.
- staged selector now reaches into framework / benchmark / narrow operational preprocessing choices while continuing to refresh compile/tree preview after every step.
- staged selector now also reaches evaluation / output / stat-test / importance choices in the current executable single-run subset.
- next major widening target after that is broader staged YAML-building beyond this current executable block.

- staged selector now exposes planned options like `factor_pca`, `mcs`, and `shap` as explicit non-executable branches rather than hiding them behind generic compile failures.

- model-grid and full-sweep are now surfaced explicitly in the staged selector as planned single-run extensions, not generic blocked states.
