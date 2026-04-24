# Layer 2 And Layer 3 Sweep Contract

Date: 2026-04-24

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

- model family;
- benchmark family;
- direct versus iterated forecast generation;
- forecast object;
- training window and refit protocol;
- validation and hyperparameter search;
- model-order selection when the order is estimator behavior;
- convergence, seed, cache, checkpointing, and execution backend.

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

The full recipe grammar already supports `sweep_axes` on Layer 2 and Layer 3.
`compile_sweep_plan()` expands those axes into Cartesian variants. Each variant
is then compiled and executed as a normal single-path recipe.

Free representation sweep means the following:

1. A researcher may place operational Layer 2 axes in `2_preprocessing.sweep_axes`.
2. A researcher may place operational Layer 3 axes in `3_training.sweep_axes`.
3. The runner materializes each Layer 2 x Layer 3 cell as a separate variant.
4. The compiler validates each cell after expansion, because some choices are
   only meaningful together.
5. Execution writes per-cell artifacts under the variant directory.
6. The study manifest records the axis values, execution status, metrics, and
   block provenance for every cell.

For example, a full recipe may sweep:

```yaml
path:
  2_preprocessing:
    sweep_axes:
      x_lag_feature_block: [none, fixed_x_lags]
      target_lag_block: [none, fixed_target_lags]
      temporal_feature_block: [none, moving_average_features, rolling_moments]
      rotation_feature_block: [none, moving_average_rotation]
      scaling_policy: [none, standard, robust]
  3_training:
    sweep_axes:
      model_family: [ridge, lasso, randomforest]
```

The compiler should not treat this as a special "preprocessing sweep" route.
It should treat this as a set of concrete `Z` construction recipes crossed with
forecast-generator choices.

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
transforms. They require `contemporaneous_x_rule=forbid_contemporaneous` so the
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

The current runtime opens the first, second, and third modes:

- `marx_replace_x_basis`: raw-panel `Z` is the MARX basis itself;
- `marx_append_to_x`: MARX features are concatenated as named blocks with base
  X, fixed X lags, or deterministic temporal blocks when
  `feature_block_combination=append_to_base_x` or
  `concatenate_named_blocks`;
- `marx_then_factor`: replace the X lag-polynomial basis with MARX features,
  then fit `pca_static_factors` on that rotated basis. Target lags may still
  be concatenated after factor scores through the existing target-lag path.

`factor_then_marx` remains gated until the runtime has an explicit factor-score
history contract for applying lag-polynomial rotations after factor extraction.

### Custom Blocks

Custom blocks require a stricter contract than the existing broad
`custom_preprocessor` hook. A custom block must return named train/pred frames,
fit-state provenance, and leakage metadata. The broad custom preprocessor may
remain as a fixed hook, but it should not be treated as a freely composable
feature block until it satisfies the block contract.

## Layer 3 Obligations

Layer 3 should receive one Layer 2 handoff bundle for every supported tabular
runtime slice. Today that bundle is `Layer2Representation`, which carries:

1. `Z_train`, `y_train`, `Z_pred`;
2. `feature_names`, `block_order`, `block_roles`;
3. `alignment`, `fit_state`, and leakage/runtime provenance.

Supported direct raw-panel paths and supported autoregressive target-lag paths
already enter Layer 3 through this same handoff shape. Sequence/tensor-style
representations are future work and remain outside the current Layer 2 closure.

The consumer contract is:

```text
fit(model_family, representation, training_spec) -> y_pred
```

Layer 3 should not branch on `feature_builder`, `x_lag_feature_block`,
`factor_feature_block`, or any other Layer 2 representation axis. Those choices
must be resolved inside Layer 2 before Layer 3 receives the matrix.

Layer 3 still owns forecast-type constraints. A raw-panel `Z` does not
automatically make iterated exogenous-X forecasting possible. Direct raw-panel
forecasting is operational; iterated raw-panel forecasting remains gated until
there is an exogenous-X path forecast contract.

## Current Operational Sweep Surface

The current runtime can sweep these Layer 2 choices in full recipes, subject to
the usual model compatibility constraints:

- `target_lag_block`: `none`, `fixed_target_lags`;
- `x_lag_feature_block`: `none`, `fixed_x_lags`;
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

## Still Gated

These are the next semantic composer tasks:

| Area | Why gated |
|---|---|
| Broader factor plus feature selection | Built-in factor blocks support `select_before_factor` and `select_after_factor`. Custom-block final-`Z` selection still needs explicit contracts. |
| MARX plus X-lag/temporal append modes | Open for named-block append / concatenate. |
| MAF rotation | Need factor-to-rotation composer and leakage metadata. |
| Custom temporal/rotation blocks | Need block-local callable contract. |
| Target normalization and inverse/evaluation scale | Need recursive fit/inverse state and metric-scale contract. |
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

1. Lower generic `Z` unification into Layer 2 by introducing one representation
   payload contract for raw-panel, factor-panel, and target-lag-only builders.
2. Keep Layer 3 as a thin consumer of that Layer 2 payload, with compatibility
   checks limited to model family and forecast type.
3. Open fixed target-lag composition with raw-panel and factor-panel direct
   models. This is the current patch.
4. Add feature-name and block-role artifacts for every `Z` column.
5. Add full recipe examples for Layer 2 x Layer 3 grids.
6. Add broader factor/selection composition semantics beyond static PCA.
7. Execute remaining MARX composition modes beyond `marx_then_factor`; they are
   already represented as gated composer modes in Layer 2 metadata.
8. Add custom block callable contracts.
9. Expose a safe simple-API representation sweep after the full-route contract
   is stable.
