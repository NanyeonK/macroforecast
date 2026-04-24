# Layer 2 Revision Plan

Date: 2026-04-22

Layer 2 is the research preprocessing and feature-representation layer. The
current runtime can execute fixed full preprocessing recipes through the
existing bridge, but the canonical design is broader: Layer 2 should let
researchers vary the target representation, predictor representation, feature
blocks, train-only preprocessing, and leakage discipline before Layer 3 fits a
forecast generator.

This document is the implementation plan for moving from the current bridge to
that canonical design without breaking old recipes.

## Target State

Layer 1 delivers an official empirical frame:

- `H`: level/raw-style panel after source loading, sample filtering,
  information-set filtering, release-lag handling, variable-universe selection,
  and raw-source missing/outlier repair;
- `X`: official transformed predictor frame when dataset transformations are
  requested;
- `target`: the target series plus horizon alignment inputs;
- provenance for all data availability and official-transform choices.

Layer 2 produces a representation bundle:

- `target_spec`: target construction and target-side transform metadata;
- `Z_train` and `Z_pred`: model input matrices for Layer 3;
- `feature_names`: stable names for all generated columns;
- `feature_block_metadata`: block-level provenance for each generated feature;
- `fit_state`: per-window state for imputers, scalers, selectors, factors,
  rotations, temporal features, custom preprocessors, and target transformers;
- `leakage_report`: evidence that train-only steps were fit only on the
  relevant training window, except for explicitly marked replication profiles.

Layer 3 consumes `target_spec`, `Z_train`, and `Z_pred`. It owns model family,
benchmark family, direct/iterated generation, validation, tuning, model-order
selection, runtime backend, and forecast aggregation protocols.

## Non-Goals

These changes should not happen as part of the Layer 2 revision:

- changing dataset/source identity rules;
- moving official FRED-MD/QD transformation ownership back from Layer 1;
- making simple API preprocessing sweeps public before fixed full recipes are
  proven;
- renaming legacy artifact columns such as `y_true` or `y_pred` without a
  compatibility layer;
- removing compatibility fallbacks before old `feature_builder` recipes have a
  lossless bridge.

## Current Bridge

The current executable bridge is:

- `feature_builder`, `predictor_family`, `data_richness_mode`, and
  `factor_count` express coarse feature representation;
- `x_missing_policy`, `x_outlier_policy`, `scaling_policy`,
  `additional_preprocessing`, `x_lag_creation`, `dimensionality_reduction_policy`,
  and `feature_selection_policy` execute for supported raw-panel paths;
- `horizon_target_construction`, `target_transform`, and `target_transformer`
  express target-side representation;
- `target_transform_policy`, `x_transform_policy`, `tcode_policy`,
  `representation_policy`, and `tcode_application_scope` remain compatibility
  bridge fields for the old `PreprocessContract`.

The current fixed full support surface is:

| Runtime profile | Status | Meaning |
|---|---|---|
| `dataset_tcode_only` | executable | Layer 1 applies official dataset transforms; Layer 2 adds no extra preprocessing. |
| `raw_only` | executable | Raw panel path with no official transforms and no extra preprocessing. |
| `raw_train_only_extra` | executable | Raw panel path plus train-only Layer 2 extra preprocessing. |
| `dataset_tcode_then_train_only_extra` | executable | Layer 1 official transforms first, then supported train-only Layer 2 extra preprocessing. |

The explicit feature-block grammar now participates in the first runtime
dispatch decision: execution derives the raw-panel versus autoregressive model
executor path from Layer 2 feature blocks and uses old `feature_builder` names
only as compatibility fallback. Fixed target-lag, fixed X-lag, and PCA
static-factor matrix composition now read explicit Layer 2 blocks first and use
old bridge fields only as fallback. Importance routines and custom model hook
contexts also report the block-derived feature runtime while retaining the
legacy builder name as provenance.

## Revision Principles

1. Preserve old recipe compatibility. Old names may warn later, but they should
   continue to compile until an explicit breaking-change window.
2. Separate provenance patches from runtime matrix patches. First record what a
   recipe means; change `Z` construction only in later patches with tests.
3. Keep `target` as the package term. Use `y` only for local numerical arrays
   or legacy artifact compatibility.
4. Fit any learned preprocessing state inside the relevant training window.
5. Treat named papers as presets over general primitives, not as layer names.
6. Keep Layer 3 responsible for forecast-generator scheduling and aggregation.
   Layer 2 defines target and feature representations; it does not train models.
7. Do not expose simple/public sweeps until fixed full recipe behavior is
   executable, provenance-rich, and tested.

## Patch Queue

### Patch L2-A: Compile-Time Representation Provenance

Goal: make compiled specs describe the canonical Layer 2 meaning of old bridge
recipes without changing runtime matrices.

Changes:

- add a compiled `layer2_representation_spec` or equivalent provenance payload;
- translate current bridge values into explicit feature-block intent:
  - `feature_builder=autoreg_lagged_target` ->
    `feature_block_set=target_lags_only`;
  - `feature_builder=raw_feature_panel` ->
    transformed/raw predictor panel block;
  - `feature_builder=raw_X_only` ->
    predictor panel block with no target-lag block;
  - `feature_builder=factor_pca` ->
    `feature_block_set=factor_blocks_only` plus a static factor block;
  - `feature_builder=factors_plus_AR` ->
    factor block plus target-lag block;
  - `data_richness_mode=full_high_dimensional_X` ->
    high-dimensional predictor block;
  - `data_richness_mode=selected_sparse_X` ->
    selected sparse predictor block;
- add target-representation provenance for `horizon_target_construction`,
  `target_transform`, `target_transformer`, and `evaluation_scale`;
- write this provenance to manifests and debug artifacts.

Acceptance:

- old recipes compile to the same execution status as before;
- prediction values do not change;
- manifest contains the new Layer 2 provenance;
- tests cover each current `feature_builder` bridge value;
- docs state that this is provenance-only;
- manifests write preferred `compatibility_source` provenance while keeping
  `source_bridge` as a compatibility alias for existing readers.

### Patch L2-B: Compatibility Name Cleanup

Goal: remove ambiguous public names before widening runtime behavior.

Changes:

- keep legacy `y_lag_count` accepted for AR/model-order selection in Layer 3;
- introduce target-language config names for Layer 2 target-lag features, such
  as `target_lag_count`, `target_lag_max`, and `target_lag_selection`;
- split `factor_ar_lags` into:
  - Layer 2 target-lag feature depth, recorded as `target_lag_count`;
  - Layer 2 factor-lag feature depth, recorded as `factor_lag_count` on the
    factor feature block;
  - Layer 3 model-order config when the lag is an estimator behavior;
- keep alias handling for existing recipes;
- update docs and examples so new prose never uses `y` for public package
  concepts.

Acceptance:

- old `y_lag_count` recipes still compile;
- new target-language config names compile and appear in manifests;
- no user-facing docs describe public Layer 2 choices as `y` choices except
  when documenting legacy compatibility;
- tests prove alias canonicalization.

### Patch L2-C: Direct Target Construction Runtime

Goal: operationalize single-target Coulombe-style direct target constructions
before path-average execution.

Candidate values:

- `average_growth_1_to_h`;
- `average_difference_1_to_h`;
- `average_log_growth_1_to_h`.

Changes:

- extend `build_horizon_target()` to construct one training target vector for
  each direct average value;
- define inverse/evaluation semantics for each direct construction;
- record target construction scale and raw-level preservation columns;
- add tests for horizon alignment, trailing missing rows, inverse behavior, and
  metric scale.

Acceptance:

- each direct average target compiles as operational only after runtime support
  exists;
- tests show that `h=1` reduces to the matching one-step construction where
  applicable;
- generated predictions preserve enough level-scale information for evaluation.

### Patch L2-D: Path-Average Target Protocol

Goal: represent path-average target construction without confusing Layer 2 and
Layer 3 responsibilities.

Candidate values:

- `path_average_growth_1_to_h`;
- `path_average_difference_1_to_h`;
- `path_average_log_growth_1_to_h`.

Layer 2 responsibilities:

- define the list of stepwise target constructions for steps `1..h`;
- expose target representation metadata for each step;
- validate that the requested construction has a supported aggregation rule.

Layer 3 responsibilities:

- schedule the stepwise forecast-generator fits;
- produce one prediction per step;
- aggregate stepwise predictions into the horizon-level forecast object;
- record per-step artifacts and aggregate artifacts.

Acceptance:

- path-average values remain `registry_only` until both Layer 2 target specs and
  Layer 3 multi-step execution are wired;
- tests prove that Layer 2 can build the stepwise target specs without fitting
  models;
- compiler tests prove that manifests record the Layer 2 protocol and the Layer
  3 execution gate. Runtime tests for per-step and aggregate artifacts belong to
  the later Layer 3 implementation patch.

### Patch L2-E: Target-Lag And X-Lag Blocks

Status: complete for fixed target and fixed predictor lag blocks. Advanced
IC/CV/custom lag-selection blocks remain registry-only until they have a
dedicated train-window fit/apply path. Fixed target-lag blocks can now compose
with raw-panel X blocks, fixed X lags, and static PCA factor blocks.

Goal: implement the first explicit feature blocks while still supporting the
old `feature_builder` bridge.

Changes:

- implement `target_lag_block=fixed_target_lags`;
- implement `x_lag_feature_block=fixed_x_lags`;
- define stable feature names, such as `target_lag_1` and
  `{predictor}_lag_{k}`;
- enforce train-window alignment and no lookahead;
- make the old bridge lower to these blocks internally where possible.

Acceptance:

- old autoregressive and raw-panel recipes keep using the same compatibility
  runtime bridge where possible;
- new explicit block recipes can compile in full mode;
- fixed target-lag plus X-lag composition executes through the raw-panel
  composer and records the target-lag block in `Z`;
- leakage tests cover origin alignment for `Z_train` and `Z_pred`.

### Patch L2-F: Factor And Selection Blocks

Status: complete for `factor_feature_block=pca_static_factors` in the supported
runtime slice, including `select_before_factor` composition with
`feature_selection_policy`. Matrix composition reads the explicit factor block
first and uses old factor/dimensionality-reduction bridge fields only as
fallback. Factor lags, supervised factors, custom factors, and
`select_after_factor` composition remain registry-only or `not_supported`
until the explicit block composer exists.

Goal: move factor construction from coarse runtime switches into explicit
Layer 2 blocks.

Changes:

- implement `factor_feature_block=pca_static_factors`;
- attach `factor_count` and any factor-lag config to factor-block metadata;
- keep `dimensionality_reduction_policy=pca` as compatibility bridge;
- define and then open the remaining `feature_selection_policy` interactions,
  especially `select_after_factor`;
- block unsupported combinations with precise compiler messages.

Acceptance:

- factor extraction is fit recursively or train-only;
- factor feature names and loadings/provenance are saved;
- old `feature_builder=factor_pca` and `factors_plus_AR` recipes translate to
  equivalent explicit blocks.

### Patch L2-G: Level, Rotation, And Temporal Blocks

Goal: support broader macro-forecasting research designs as combinations of
general feature primitives.

Candidate blocks:

- `level_feature_block`: target level add-back, selected level add-backs,
  level-growth pairs;
- `rotation_feature_block`: MARX, MAF, moving-average rotations, custom
  rotations;
- `temporal_feature_block`: moving-average features, rolling moments, local
  temporal factors, volatility features.

Changes:

- implement one block family at a time;
- require train-window fit/apply tests for any learned rotation;
- use Goulet Coulombe et al. and MARX as reference presets, not as special
  layer names;
- keep custom hooks for researchers whose preprocessing is outside built-ins.

Current lowered slice:

- `level_feature_block=target_level_addback`, `x_level_addback`,
  `selected_level_addbacks`, and `level_growth_pairs` are executable for
  raw-panel feature runtimes.
  Target add-back appends the observed target level at the feature row date and
  at the prediction origin. X-level add-back appends raw-level `H` predictor
  values preserved after Layer 1 raw missing/outlier handling and before
  official transforms/T-codes. Selected level add-back applies the same
  source/alignment rule to `leaf_config.selected_level_addback_columns`.
  Level-growth pairs record existing transformed predictor columns with
  raw-level counterparts from `leaf_config.level_growth_pair_columns`. These
  values reject contemporaneous-oracle X alignment because that would require
  target-date information at prediction time.
- `temporal_feature_block=moving_average_features`, `rolling_moments`,
  `local_temporal_factors`, and `volatility_features` are executable for
  raw-panel feature runtimes. They append trailing 3-period moving averages,
  mean/variance moments, deterministic local temporal factors, or rolling
  volatility of the base predictor columns using only information available
  through each row date / prediction origin, and reject X-lag/factor bridge
  composition for now. Local temporal factors are row-wise cross-sectional
  mean/dispersion summaries of the active predictor panel with trailing time
  smoothing, not learned PCA/static factors.
- `temporal_feature_block=custom_temporal_features` remains registry-only by
  design. It should not silently reuse the broad `custom_preprocessor` hook:
  operational custom temporal blocks need a block-local callable contract that
  returns train/pred feature frames, stable feature names, fit-state
  provenance, and leakage metadata.
- Deterministic raw-panel append blocks now compose without an additional
  block-composer gate. Fixed X lags are constructed first, temporal append
  blocks are added next, `moving_average_rotation` append blocks are added
  after that, and level add-backs keep their existing final append position.
  The feature-name order mirrors this runtime order.
- `rotation_feature_block=none`, `moving_average_rotation`, and
  `marx_rotation` are executable for raw-panel feature runtimes. `none` records
  explicit no-rotation provenance when selected. `moving_average_rotation`
  appends deterministic trailing 3- and 6-period moving-average rotations of
  each active predictor column with `{predictor}_rotma3` /
  `{predictor}_rotma6` public names, using only information available through
  each row date / prediction origin. It can compose with fixed X lags and
  deterministic temporal append blocks. `marx_rotation` requires
  `leaf_config.marx_max_lag`, builds the cumulative moving-average
  lag-polynomial basis, replaces the source X lag-polynomial basis in final
  `Z`, and now supports `marx_then_factor` with `pca_static_factors`.
- `rotation_feature_block=maf_rotation` and `custom_rotation` remain
  registry-only. The compiler still records explicit boundary metadata when
  those values are selected: MAF requires factor-to-rotation composition, and
  custom rotations require a block-local callable contract. These values should
  not silently reuse `moving_average_rotation` or the broad `custom_preprocessor`
  hook.
- MARX now has an executable code-level contract,
  `lag_polynomial_rotation_contract_v1`. It fixes naming
  (`{predictor}_marx_ma_lag1_to_lag{p}` / `{predictor}__marx_ma_lag1_to_lag{p}`),
  feature order (predictor-major, then rotation order), alignment
  (`Z_{i,p,t} = p^{-1} * sum_{j=1}^{p} X_{i,t-j}`), and basis composition
  (`replace_lag_polynomial_basis`). MARX-to-factor composition is now explicit
  as `marx_then_factor`: the runtime first builds the MARX basis, then fits
  static PCA factors on that rotated basis. External X-lag append, temporal
  append, and remaining MARX composition modes remain gated.

Acceptance:

- each block writes block-level metadata;
- each block can be disabled independently;
- block composition is deterministic and feature names are stable;
- unsupported combinations fail at compile time.

### Patch L2-H: Bridge Dispatch Retirement

Goal: make the explicit block grammar the runtime path while preserving old
recipe compatibility.

Changes:

- route runtime `Z` construction through an explicit feature-block builder;
- lower old bridge fields to explicit blocks in the compiler;
- mark old bridge names as compatibility aliases in docs and manifests;
- keep deprecation warnings separate from behavior changes.

Acceptance:

- old recipes and new explicit-block recipes produce equivalent predictions for
  supported cases;
- manifests show both canonical block specs and legacy source fields;
- full test suite passes with no matrix drift outside intentionally changed
  runtime paths.

### Patch L2-I: Full Sweeps And Simple API Exposure

Goal: expose sweeps only after fixed full recipes are stable.

Changes:

- allow full-mode sweeps over explicitly supported Layer 2 axes;
- keep simple API defaults conservative;
- add compiler guards for explosive or unsupported block combinations;
- document which axes are fixed-only, sweep-safe, or research-experimental.

Acceptance:

- sweep manifests record every varied Layer 2 axis;
- unsupported combinations are blocked before execution;
- simple API does not expose registry-only or path-average values prematurely.

## Test Matrix

Every runtime patch should include:

- compile tests for old bridge recipes and new explicit recipes;
- train-window leakage tests for every learned transformation;
- prediction equivalence tests for bridge-lowered recipes where behavior should
  not change;
- manifest/provenance tests for `target_spec`, feature blocks, and fit state;
- horizon alignment tests for target-side constructions;
- docs build.

For target construction patches, also test:

- `h=1` equivalence where mathematically applicable;
- non-positive target rejection for log constructions;
- raw-level preservation columns;
- metric-scale behavior.

For feature-block patches, also test:

- `Z_train`/`Z_pred` row alignment;
- deterministic feature ordering;
- stable feature names;
- behavior with leading/mid-sample missing values after Layer 1.

## Status Tracking

| Slice | Status | Notes |
|---|---|---|
| Terminology cleanup | done | `target` is canonical; legacy `y_*` artifacts remain compatible. |
| Feature-block grammar | done | Explicit blocks now drive the first raw-panel/autoregressive runtime dispatch decision. |
| Compile-time provenance | done | Compiled and runtime manifests record `layer2_representation_spec`; runtime matrices are unchanged. |
| Compatibility name cleanup | done | Added `target_lag_selection`, `target_lag_count`, and `factor_lag_count` metadata while keeping legacy `y_lag_count` / `factor_ar_lags` accepted for old recipes. New compiled specs no longer emit `factor_ar_lags` in `training_spec`. |
| Direct target constructions | done | Direct average growth/difference/log-growth values compile and execute with construction-scale metrics plus level-scale preservation columns. |
| Path-average target constructions | done, protocol-only | Layer 2 stepwise target protocol is recorded; execution remains gated until Layer 3 multi-step fit/aggregation lands. |
| Explicit target/X lag blocks | done for fixed blocks | Fixed target-lag and fixed X-lag matrix composition now read `target_lag_block` / `x_lag_feature_block` before old bridge fields; fixed target-plus-X composition is executable in raw-panel direct runtimes. |
| Factor/selection blocks | done for static PCA + explicit before/after semantics | PCA static-factor matrix composition now reads `factor_feature_block` before old factor/dimred bridges; `feature_selection_policy` can compose as `select_before_factor` or `select_after_factor` in the supported static-PCA slice. |
| Level/rotation/temporal blocks | done for built-ins | Level blocks, deterministic temporal blocks, moving-average rotation, and MARX lag-polynomial rotation are executable for raw-panel builders; MAF/custom and semantic cross-block composition remain gated as future feature work. |
| Bridge dispatch retirement | done for supported runtime slices | Executor-family dispatch, fixed target/X-lag matrix composition, PCA static-factor matrix composition, target-transformer gates, importance artifacts, custom hook contexts, and decomposition component naming now route through explicit Layer 2 block/runtime provenance. |
| Representation handoff unification | done for supported runtime slices | Supported raw-panel and autoregressive target-lag runtimes now emit a canonical `Layer2Representation` bundle with `Z_train`, `y_train`, `Z_pred`, feature names, block roles, alignment, fit-state provenance, and leakage metadata before Layer 3 fit/predict. |
| Simple/public sweeps | out of L2 cleanup scope | Fixed full recipes are closed under the supported runtime surface. Public sweep exposure is a separate API/governance step, not a Layer 2 boundary blocker. |

## Layer 2 Closure

Layer 2 is complete for the current migration target: fixed full recipes and
existing simple defaults compile and execute through canonical Layer 2
feature-representation contracts where runtime support exists. Compatibility
aliases remain intentionally accepted for old recipes and old manifests.

The remaining items in this file are not cleanup blockers:

- broader factor/selection composition beyond static PCA;
- MARX with additional X-lag/temporal/factor composition;
- MAF/custom rotations;
- custom temporal/feature-block callable contracts;
- target-side normalization/inverse/evaluation-scale expansion;
- public Layer 2 sweep exposure.

Those are semantic feature-composer or API-governance tasks. They should start
from explicit contracts and acceptance tests rather than from bridge cleanup.
