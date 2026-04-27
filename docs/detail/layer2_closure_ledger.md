# Layer 2 Closure Ledger

Date: 2026-04-24

Layer 2 is the research preprocessing and feature-representation layer. It
constructs the target-side representation and the final feature matrix `Z`
handed to Layer 3.

This ledger separates three states:

- operational: executable in the current runtime slice;
- contract-defined gated: compiler/runtime metadata is explicit, but execution
  remains gated until fit/apply or Layer 3 support exists;
- compatibility-only: accepted to keep old recipes working, but not the primary
  language for new recipes.

## Operational

Target construction:

- `future_target_level_t_plus_h`
- `future_diff`
- `future_logdiff`
- `average_growth_1_to_h`
- `average_difference_1_to_h`
- `average_log_growth_1_to_h`

Target-side scale:

- `target_transform=level`, `difference`, `log`, `log_difference`,
  `growth_rate`
- `target_normalization=none`, `zscore_train_only`, `robust_zscore`,
  `minmax`, `unit_variance`
- `inverse_transform_policy=none`, `target_only`, `forecast_scale_only`
- `evaluation_scale=raw_level` / `original_scale`, `transformed_scale`,
  `both`
- custom `target_transformer` under its existing raw-scale runtime gates

Feature blocks:

- `target_lag_block=none`, `fixed_target_lags`
- `x_lag_feature_block=none`, `fixed_predictor_lags`
- `factor_feature_block=none`, `pca_static_factors`
- `feature_selection_semantics=select_before_factor`, `select_after_factor`
  with static PCA factor blocks
- `level_feature_block=target_level_addback`, `x_level_addback`,
  `selected_level_addbacks`, `level_growth_pairs`
- `temporal_feature_block=moving_average_features`, `rolling_moments`,
  `local_temporal_factors`, `volatility_features`
- `rotation_feature_block=moving_average_rotation`, `marx_rotation`
- MARX basis replacement, `marx_then_factor`, and named-block append
  composition with fixed X lags or deterministic temporal blocks via
  `feature_block_combination=append_to_base_predictors` / `concatenate_named_blocks`
- `factor_feature_block=pca_factor_lags`, `supervised_factors`
- registered `custom_feature_block_callable_v1` temporal, rotation, and factor
  blocks
- `feature_block_combination=replace_with_selected_blocks`, `append_to_base_predictors`,
  `append_to_target_lags`, `concatenate_named_blocks`

## Contract-Defined Gated

Target-side scale still gated:

- `inverse_transform_policy=custom`

The compiler records `target_scale_contract_v1` in
`layer2_representation_spec.target_representation.target_scale_contract`.
Built-in target normalization is now fit per training window, and prediction
artifacts carry model, transformed, and original scale columns plus dual-scale
metric summaries where requested.

Custom feature methods without a registered callable:

- `temporal_feature_block=custom_temporal_features`
- `rotation_feature_block=custom_rotation`
- `factor_feature_block=custom_factors`
- `feature_block_combination=custom_feature_combiner`

Block-local values use `custom_feature_block_callable_v1`. A registered
callable is executable when the recipe names it through `custom_feature_blocks`
or the block-specific field such as `custom_temporal_feature_block`.
`feature_block_combination=custom_feature_combiner` uses
`custom_feature_combiner_v1` and is executable when the recipe names a
registered `custom_feature_combiner`. The existing `custom_preprocessor` hook
is broader and does not replace these Layer 2 representation contracts.

Custom-block final-`Z` selection:

- `feature_selection_semantics=select_after_custom_feature_blocks`

This records `custom_final_z_selection_v1` and is executable when an
operational feature-selection policy is applied after registered custom blocks
or a registered custom combiner.

MARX and factor frontier:

- `factor_then_marx` and `maf_rotation` are operational for
  `pca_static_factors` through `factor_rotation_order=factor_then_rotation`.
- feature selection after custom append blocks

Custom append-block selection remains represented as an explicit gated composer
mode, not a silent alias.

Layer 3 handoff:

- path-average target construction choices record
  `path_average_target_protocol_v1`; Layer 3 now executes the protocol through
  stepwise fit/predict/aggregate runtime and writes `path_average_steps.csv`.

## Compatibility-Only

The following names remain accepted as old recipe inputs or manifest aliases:

- `feature_builder`
- `source_bridge`
- `y_lag_count`
- `factor_ar_lags`
- `dimensionality_reduction_policy` as a fallback for explicit
  `factor_feature_block`
- old Layer 2 T-code bridge fields now owned by Layer 1 official-transform
  axes

New docs and generated recipes should prefer:

- `target_lag_block`
- `x_lag_feature_block`
- `factor_feature_block`
- `level_feature_block`
- `temporal_feature_block`
- `rotation_feature_block`
- `feature_selection_semantics`
- `compatibility_source`

## Sweep Governance

The sweep planner may Cartesian-expand all requested Layer 2 representation
axes. It records `sweep_governance_v1` with:

- swept Layer 2 axes;
- swept representation axes;
- whether model and representation axes are co-swept;
- the policy that invalid combinations are materialized, then gated at variant
  compile or execution time.

This keeps research exploration broad while preserving per-variant provenance
and failure visibility.

## Next Build Order

This is no longer cleanup work. It is the semantic composer backlog for new
research features.

Detailed interface contracts and acceptance tests for these items are fixed in
`layer2_layer3_detailed_design.md`. That document should be updated before a
future item moves from `contract-defined gated` to `operational`.

1. Target-side `inverse_transform_policy=custom`.
   This needs a callable inverse contract plus metric-scale and artifact
   guarantees.
2. Extended path-average variants beyond the current point-forecast runtime.
   Stepwise point forecasts are executable; interval, density, and custom
   target-transformer variants still need their own payload contracts.
4. Sequence/tensor representation handoff.
   Current tabular `Layer2Representation` is closed for supported raw-panel,
   factor, custom-block, and target-lag runtimes. Sequence/tensor models need a
   separate handoff contract before they should enter full grids.
