# Layer 2 And Layer 3 Sweep Contract

Date: 2026-04-24

Detailed forward design for the next Layer 2 representation composers and
Layer 3 forecast-generation protocols lives in
`layer2_layer3_detailed_design.md`. This page defines the sweep contract and
runtime audit semantics for the currently supported grid machinery.

This document defines the target contract for freely sweeping research feature
representations together with forecast generators. It is the bridge between
Layer 2, which constructs the research representation `Z`, and Layer 3, which
fits forecast generators on the Layer 2 handoff.

## Core Goal

Researchers should be able to sweep over operational Layer 2 representation
choices and operational Layer 3 training choices without hand-writing special
case recipes for each combination.

The goal is not to mark every named registry value executable immediately.
The goal is stronger and more precise:

- every value marked `operational` must either compose with other operational
  values through a declared contract, or declare a narrow semantic
  incompatibility;
- every sweep variant must record the exact Layer 2 representation and Layer 3
  forecast-generator contract that produced it;
- unsupported values must remain `registry_only` or `not_supported_yet`, not
  silently enter a sweep;
- failed cells in large exploratory grids must be resumable and auditable when
  the selected Layer 0 failure policy allows partial results.

## Boundary

Layer 2 owns the transformation from Layer 1 outputs to model inputs:

- official-frame target series and predictor panel from Layer 1;
- post-frame X conditioning;
- target representation and horizon target construction;
- target-history blocks;
- predictor-lag blocks;
- factor blocks;
- level add-back blocks;
- temporal feature blocks;
- rotation blocks;
- feature-selection blocks;
- feature-block combination into `Z_train` and `Z_pred`;
- the unified representation payload handed to forecast generators;
- block-local leakage and fit-state provenance.

Layer 3 owns forecast generation:

- forecast generator family, currently exposed as compatibility axis
  `model_family`;
- baseline generator role assignment, currently exposed as compatibility axis
  `benchmark_family`;
- direct versus iterated forecast generation;
- forecast object;
- training window and refit protocol;
- validation and hyperparameter search;
- model-order selection when the order is estimator behavior;
- estimator training discipline: convergence, early stopping, and
  model-specific seed use under the Layer 0 reproducibility policy.

Layer 0 owns broad run control such as failure policy, compute mode,
reproducibility mode, and broad cache/checkpoint policy. Layer 1 owns data
timing discipline such as vintage, release lag, contemporaneous information,
and availability.

Layer 3 must not decide which feature blocks exist. It consumes the Layer 2
representation payload plus training settings.

## Terms

`H`
: The Layer 1 official frame after source loading, official transform policy,
  raw-source missing/outlier decisions, release-lag policy, and target/X
  alignment.

`X`
: The active predictor panel selected from `H` by Layer 2 predictor-family and
  representation rules.

`target history`
: Target observations available at the forecast origin. For a direct row with
  forecast origin `t`, `target_lag_1` means the target value observed at `t`,
  `target_lag_2` means `t-1`, and so on.

`Z`
: The final feature matrix handed to Layer 3.

`block`
: A named Layer 2 feature source. Examples: base transformed X, fixed target
  lags, fixed X lags, PCA factors, level add-backs, moving-average features,
  rolling moments, MARX rotations.

`composer`
: The Layer 2 runtime object or function that builds one or more blocks,
  aligns their train/prediction rows, applies fit discipline, and returns
  `Z_train`, `Z_pred`, feature names, and provenance.

## Sweep Model

The full recipe grammar supports `sweep_axes` on Layer 2 and Layer 3.
`compile_sweep_plan()` expands those axes into Cartesian variants. It also
supports `leaf_sweep_axes` for variant-specific `leaf_config` values such as
registered custom feature-block names, and `nested_sweep_axes` for conditional
method grids. Each variant is then compiled as a normal single-path recipe.
Executable variants run through `execute_recipe`; non-executable cells are
reported before execution.

Free representation sweep means the following:

1. A researcher may place operational Layer 2 axes in `2_preprocessing.sweep_axes`.
2. A researcher may place operational Layer 3 axes in `3_training.sweep_axes`.
3. A researcher may place custom method names or configs in
   `leaf_sweep_axes`; these materialize into variant `leaf_config`.
4. A researcher may use `nested_sweep_axes` to bind a custom name only to the
   parent custom axis value that needs it.
5. The runner materializes each Layer 2 x Layer 3 cell as a separate variant.
6. The compiler validates each cell after expansion, because some choices are
   only meaningful together.
7. With `failure_policy=skip_failed_cell`, compile-invalid cells are marked
   `skipped` before execution. The variant directory still receives
   `compiler_manifest.json` for audit.
8. Executed cells write per-cell artifacts under the variant directory.
9. The study manifest records the axis values, execution status, metrics,
   compiler status, Layer 3 capability cell, and block provenance for every
   cell.

For example, a full recipe may sweep:

```yaml
path:
  2_preprocessing:
    sweep_axes:
      x_lag_feature_block: [none, fixed_predictor_lags]
      target_lag_block: [none, fixed_target_lags]
      temporal_feature_block: [none, moving_average_features, rolling_moments]
      rotation_feature_block: [none, moving_average_rotation]
      scaling_policy: [none, standard, robust]
  3_training:
    sweep_axes:
      model_family: [ridge, lasso, random_forest]
```

`model_family` is the current recipe spelling for the candidate
`forecast_generator_family`. `benchmark_family` is the current recipe spelling
for the baseline generator role assignment. They are kept as compatibility
axes while the canonical docs use forecast-generator language.

The compiler should not treat this as a special "preprocessing sweep" route.
It should treat this as a set of concrete `Z` construction recipes crossed with
forecast-generator choices.

Custom method names are not registry axes. They are variant leaf configuration.
Use `leaf_sweep_axes` when every variant should receive the key:

```yaml
path:
  2_preprocessing:
    leaf_sweep_axes:
      custom_feature_combiner: [my_combiner_a, my_combiner_b]
```

Use `nested_sweep_axes` when the custom name is meaningful only for a custom
parent value:

```yaml
path:
  2_preprocessing:
    nested_sweep_axes:
      temporal_feature_block:
        moving_average_features: {}
        custom_temporal_features:
          leaf_config.custom_temporal_feature_block:
            - my_temporal_block
            - my_second_temporal_block
  3_training:
    sweep_axes:
      model_family: [ridge, my_custom_generator]
```

The example above creates one built-in temporal-feature variant and two custom
temporal-feature variants, then crosses them with the selected forecast
generators. The built-in temporal-feature variant does not receive an unused
custom block name.

Invalid cells are expected in broad research grids. For example, a raw-panel
feature runtime crossed with `model_family=ar` is a Layer 2 x Layer 3
compatibility failure, not a crash. The sweep manifest records this as:

- variant `status="skipped"`;
- `compiler_status="blocked_by_incompatibility"` or `not_supported`;
- `compiler_blocked_reasons`;
- `layer3_capability_cell`, copied from the compiled manifest;
- root summary counts: `successful`, `failed`, `skipped`, `invalid_cells`, and
  `runnable_variants`.

## Layer 2 Representation Contract

Every operational Layer 2 composer must produce one unified handoff payload
with these fields:

| Field | Meaning |
|---|---|
| `Z_train` | 2-D numeric array or frame for the current recursive training window. |
| `y_train` | 1-D numeric target vector aligned to `Z_train`. |
| `Z_pred` | 2-D numeric one-row prediction matrix for the forecast origin. |
| `feature_names` | Stable public names in the same order as `Z_train` columns. |
| `block_order` | Ordered list of blocks used to form `Z`. |
| `block_roles` | Map from feature name to block role: base X, target lag, factor, level, temporal, rotation, selected X, custom. |
| `fit_state` | Recursive fit metadata, such as scaler, selector, factor loadings, or custom block provenance. |
| `alignment` | Train-row and prediction-row timing rules for each block. |
| `leakage_contract` | Whether the block uses only forecast-origin information. |

The initial runtime still uses functions rather than a dedicated dataclass, but
new implementation should move toward this payload shape. This unification is a
Layer 2 ownership item: Layer 2 must resolve raw-panel, factor-panel, and
target-lag-only construction into the same public handoff before Layer 3 sees
the data.

## Block Semantics

### Base X

Base X starts from the active predictor panel. It may be raw official frame X,
dataset t-code transformed X, or post-frame extra-preprocessed X depending on
the Layer 1 official-transform choices and Layer 2 frame-conditioning choices.

### Target Lags

Fixed target lags are origin-aligned. For a direct row whose forecast origin is
`t`, `target_lag_1` is `target_t`, `target_lag_2` is `target_{t-1}`. For the
prediction row at origin `T`, the same rule gives `target_T`, `target_{T-1}`.

Target lags are allowed to compose with raw-panel/factor-panel features when
the runtime can concatenate them after block-local conditioning. This is not
Layer 3 AR model-order selection. AR-BIC model-order selection remains a
Layer 3 estimator behavior.

### X Lags

Fixed X lags are origin-aligned predictor lags. The current operational block
uses one lag. Leading unavailable values are zero-filled, matching the existing
runtime behavior.

### Factor Blocks

`pca_static_factors` is a train-window factor block. It fits PCA on the current
training window and transforms the prediction row with the same loadings.
Factors must carry source feature names, loadings, number of components, and
window metadata.

When target lags are also present, the default rule is:

- fit PCA on the X-side block only;
- concatenate target lags after factor scores;
- do not let target lags enter the PCA unless a future explicit combiner says
  `factor_of_augmented_panel`.

### Feature Selection

Feature selection now supports one explicit factor-composer semantic and still
reserves the second:

- `select_before_factor`: select X columns, then estimate factor block;
- `select_after_factor`: estimate factor block, then select among final `Z`
  columns.

The current runtime opens both semantics for
`factor_feature_block=pca_static_factors` and the equivalent
`dimensionality_reduction_policy` bridge. `select_before_factor` performs
selection on the raw X-side panel before factor extraction.
`select_after_factor` performs selection on the final composed `Z` after an
executable built-in factor block plus any appended target-lag,
deterministic-component, or structural-break columns.

### Level Blocks

Level blocks add raw-level values preserved from Layer 1 before official
transforms. They require `contemporaneous_x_rule=forbid_same_period_predictors` so the
feature row uses information available at the forecast origin.

### Temporal Blocks

Temporal blocks are local deterministic time-series summaries of active
predictors, such as moving averages, rolling moments, volatility features, and
local temporal cross-sectional summaries. They can compose with fixed X lags
and moving-average rotation in the current raw-panel runtime.

### Rotation Blocks

`moving_average_rotation` appends deterministic trailing moving-average
rotations. `marx_rotation` replaces the source X lag-polynomial basis. Because
MARX is a replacement basis rather than a simple append block, composition must
distinguish:

- `marx_replace_x_basis`;
- `marx_append_to_x`;
- `marx_then_factor`;
- `factor_then_marx`.

The current runtime opens all four modes:

- `marx_replace_x_basis`: raw-panel `Z` is the MARX basis itself;
- `marx_append_to_x`: MARX features are concatenated as named blocks with base
  X, fixed X lags, or deterministic temporal blocks when
  `feature_block_combination=append_to_base_predictors` or
  `concatenate_named_blocks`;
- `marx_then_factor`: replace the X lag-polynomial basis with MARX features,
  then fit `pca_static_factors` on that rotated basis. Target lags may still
  be concatenated after factor scores through the existing target-lag path.
- `factor_then_marx`: fit `pca_static_factors` on the X-side panel, transform
  the full start-to-origin factor-score history with train-window loadings,
  then apply the MARX lag-polynomial basis to those factor scores.

`maf_rotation` is also operational when paired with `pca_static_factors`. It is
not a raw-X moving-average append; it is a factor-score rotation that first
fits static factor histories and then builds trailing moving-average factor
features. Layer 2 records this choice through `factor_rotation_order` and the
`factor_score_history_contract_v1` fit-state metadata.

### Custom Blocks

Custom feature blocks use `custom_feature_block_callable_v1`. A registered
custom block must return named train/pred frames, feature names, runtime feature
names, fit-state provenance, leakage metadata, and block provenance. Registered
custom temporal, rotation, and factor blocks are executable through the Layer 2
custom-block slots.

The broad `custom_preprocessor` hook remains a post-representation matrix hook.
It can be fixed in a recipe, but it should not be treated as a freely
composable feature block unless it is expressed through the custom feature
block contract.

## Layer 3 Obligations

Layer 3 should receive one Layer 2 handoff bundle for every supported tabular
runtime slice. Today that bundle is `Layer2Representation`, which carries:

1. `Z_train`, `y_train`, `Z_pred`;
2. `feature_names`, `block_order`, `block_roles`;
3. `alignment`, `fit_state`, and leakage/runtime provenance.

Supported direct raw-panel paths and supported autoregressive target-lag paths
already enter Layer 3 through this same handoff shape. Sequence/tensor-style
representations are future work and remain outside the current Layer 2 closure.
The built-in raw-panel factor-model adapters (`pcr`, `pls`, and
`factor_augmented_linear`) now also consume that handoff, so factor, rotation,
level, temporal, and target-lag block composition is resolved before those
models fit.

The consumer contract is:

```text
fit(forecast_generator_family, representation, training_spec) -> y_pred
```

Layer 3 should not branch on `feature_builder`, `x_lag_feature_block`,
`factor_feature_block`, or any other Layer 2 representation axis. Those choices
must be resolved inside Layer 2 before Layer 3 receives the matrix.

Layer 3 still owns forecast-type constraints. A raw-panel `Z` does not
automatically make iterated exogenous-X forecasting possible. Direct raw-panel
forecasting is operational; iterated raw-panel forecasting remains gated until
there is an exogenous-X path forecast contract.

The compiler now records these gates as `layer3_capability_matrix` in every
new manifest. The matrix is not a Layer 2 representation spec. It is a Layer 3
runtime support table over `model_family`, resolved Layer 2 `feature_runtime`,
`forecast_type`, and `forecast_object`, plus an `active_cell` for the current
recipe. Full sweep runners should use the same matrix to report or prune
invalid Layer 2 x Layer 3 cells.

The current manifest keys remain compatibility names. Their canonical meaning
is:

| Compatibility key | Canonical meaning |
|---|---|
| `model_family` | `forecast_generator_family` |
| `benchmark_family` | baseline generator role assignment |
| `feature_runtime` | `representation_runtime` |
| `forecast_type` | `forecast_protocol` |

The matrix also carries a status catalog and payload contract names. Direction,
interval, and density forecast objects are operational as typed wrappers over
the scalar generator. Reserved future cells document support targets that
remain closed until their contracts exist.

The canonical list of operational and gated contracts now lives in
`layer_contract_ledger.md`. This sweep contract should reference that ledger
rather than carrying a second contract-status table.

Current recipes still reject dropped values until the corresponding runtime
contract is implemented.

## Current Operational Sweep Surface

The current runtime can sweep these Layer 2 choices in full recipes, subject to
the usual model compatibility constraints:

- `target_lag_block`: `none`, `fixed_target_lags`;
- `x_lag_feature_block`: `none`, `fixed_predictor_lags`;
- `factor_feature_block`: `none`, `pca_static_factors`;
- `level_feature_block`: `none`, `target_level_addback`,
  `x_level_addback`, `selected_level_addbacks`, `level_growth_pairs`;
- `temporal_feature_block`: `none`, `moving_average_features`,
  `rolling_moments`, `local_temporal_factors`, `volatility_features`;
- `rotation_feature_block`: `none`, `moving_average_rotation`,
  `marx_rotation` with replacement-basis support plus `marx_then_factor`
  static-PCA composition;
- X-side frame conditioning axes already marked operational, such as missing,
  outlier, scaling, HP filter, and train-only fit scope.

This patch opens the first previously gated composition class:

- fixed target lags plus raw-panel X blocks;
- fixed target lags plus fixed X lags;
- fixed target lags plus factor blocks, where target lags are concatenated
  before or after the selected block depending on `feature_block_combination`.
- raw predictor feature selection followed by built-in factor blocks
  (`select_before_factor`) in supported raw-panel runtimes.
- built-in factor blocks followed by final-`Z` feature selection
  (`select_after_factor`), including appended target lags and deterministic
  columns.
- MARX basis replacement followed by static PCA factors
  (`marx_then_factor`) in supported raw-panel runtimes.
- static PCA factors followed by MARX factor-score rotations
  (`factor_then_marx`) or moving-average factor rotations (`factor_then_maf`).

## Still Gated

These are the next semantic composer tasks:

| Area | Why gated |
|---|---|
| Broader factor plus feature selection | Built-in factor blocks support `select_before_factor` and `select_after_factor`. Broader custom factor/selection slices must satisfy the Layer 2 method-extension contracts in `layer_contract_ledger.md`. |
| Custom combiners | Operational for registered `custom_feature_combiner_v1` slices; broader combiner families need explicit names, fit-state provenance, and invalid-cell tests. |
| Custom-block final-`Z` selection | Operational for registered supported slices through `custom_final_z_selection_v1`; broader selector families need explicit provenance over user-created columns. |
| Custom inverse policies | Need extension contracts beyond built-in target-scale and target-transformer policies. |
| Iterated raw-panel forecasting | Need exogenous-X forecasting or scenario path contract. |

## Acceptance Tests

Before a Layer 2 x Layer 3 combination is marked operational, tests must cover:

- compile status for the fixed recipe;
- sweep-plan expansion when the axis is placed in `sweep_axes`;
- runtime execution on a fixture dataset;
- stable `feature_names` order;
- no lookahead in target lag, X lag, temporal, rotation, and level blocks;
- fit-state provenance for estimated blocks;
- study-manifest recording of variant axis values;
- failure-policy behavior for invalid cells in large grids.

## Implementation Roadmap

1. Keep Layer 3 as a thin consumer of the Layer 2 representation payload, with
   compatibility checks limited to model family, forecast type, forecast
   object, and runtime support.
2. Continue removing Layer 2 compatibility fields from new `training_spec`
   generation while preserving legacy aliases for old recipes and manifests.
   The first passes moved target-lag provenance, custom hook selections, and
   factor-count configuration into `layer2_representation_spec`. The latest
   pass split legacy `factor_ar_lags` into explicit Layer 2 `target_lag_count`
   and factor-block `factor_lag_count` metadata for new compiled specs.
3. Promote reserved `layer3_capability_matrix.future_cells` to executable
   registry values only when their runtime payload, scoring, and artifact
   contracts are implemented.
4. Add full recipe examples for Layer 2 x Layer 3 grids.
5. Execute remaining semantic composer contracts:
   custom inverse policies and the sequence/tensor representation handoff.
6. Expose safe simple-API representation sweeps only after the full-route
   naming, result-summary, and invalid-cell reporting contracts are stable.
