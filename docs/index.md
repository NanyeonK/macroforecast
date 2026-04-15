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
  - first train-only raw-panel path with `x_missing_policy=em_impute` and `scaling_policy=standard`
- operational statistical tests:
  - `dm`
  - `cw`
- operational importance methods:
  - `minimal_importance`

Current roadmap focus
- next major widening target is broader importance coverage and richer evaluation/statistical-testing coverage beyond the first DM/minimal-importance slice.
