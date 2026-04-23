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
- replacing current runtime dispatch before old `feature_builder` recipes have
  a lossless bridge.

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

The explicit feature-block grammar is registry-only today. Runtime still
constructs matrices through the coarse bridge.

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
    factor block;
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
- docs state that this is provenance-only.

### Patch L2-B: Compatibility Name Cleanup

Goal: remove ambiguous public names before widening runtime behavior.

Changes:

- keep legacy `y_lag_count` accepted for AR/model-order selection in Layer 3;
- introduce target-language config names for Layer 2 target-lag features, such
  as `target_lag_count`, `target_lag_max`, and `target_lag_selection`;
- split `factor_ar_lags` into:
  - Layer 2 factor/target-lag feature dimensions, such as `factor_lag_count`
    or block-level lag config;
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
dedicated train-window fit/apply path.
Joint target-lag plus X-lag block composition remains gated until the explicit
block composer replaces the compatibility bridge.

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
- unsupported target-lag plus X-lag composition reports `not_supported`
  instead of pretending the target-lag block is in `Z`;
- leakage tests cover origin alignment for `Z_train` and `Z_pred`.

### Patch L2-F: Factor And Selection Blocks

Status: complete for `factor_feature_block=pca_static_factors` through
compatibility lowering. Factor lags, supervised factors, custom factors, and
factor/selection composition remain registry-only or `not_supported` until the
explicit block composer exists.

Goal: move factor construction from coarse runtime switches into explicit
Layer 2 blocks.

Changes:

- implement `factor_feature_block=pca_static_factors`;
- attach `factor_count` and any factor-lag config to factor-block metadata;
- keep `dimensionality_reduction_policy=pca` as compatibility bridge;
- define how `feature_selection_policy` interacts with factor blocks and raw
  predictor blocks;
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
  raw-panel feature builders.
  Target add-back appends the observed target level at the feature row date and
  at the prediction origin. X-level add-back appends raw-level `H` predictor
  values preserved after Layer 1 raw missing/outlier handling and before
  official transforms/T-codes. Selected level add-back applies the same
  source/alignment rule to `leaf_config.selected_level_addback_columns`.
  Level-growth pairs record existing transformed predictor columns with
  raw-level counterparts from `leaf_config.level_growth_pair_columns`. These
  values reject contemporaneous-oracle X alignment because that would require
  target-date information at prediction time.
- `temporal_feature_block=moving_average_features`, `rolling_moments`, and
  `volatility_features` are executable for raw-panel feature builders. They
  append trailing 3-period moving averages, mean/variance moments, or rolling
  volatility of the base predictor columns using only information available
  through each row date / prediction origin, and reject X-lag/factor bridge
  composition for now.
- `rotation_feature_block=*` and the remaining `temporal_feature_block=*`
  values remain future block-composition work.

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
| Feature-block grammar | done, registry-only | Runtime still uses the coarse bridge. |
| Compile-time provenance | done | Compiled and runtime manifests record `layer2_representation_spec`; runtime matrices are unchanged. |
| Compatibility name cleanup | done, provenance-only | Added `target_lag_selection` and `target_lag_count` provenance while keeping legacy `y_lag_count` / `factor_ar_lags` accepted. |
| Direct target constructions | done | Direct average growth/difference/log-growth values compile and execute with construction-scale metrics plus level-scale preservation columns. |
| Path-average target constructions | done, protocol-only | Layer 2 stepwise target protocol is recorded; execution remains gated until Layer 3 multi-step fit/aggregation lands. |
| Explicit target/X lag blocks | planned | First runtime block migration. |
| Factor/selection blocks | planned | PCA/static factors and selection provenance. |
| Level/rotation/temporal blocks | planned | MARX/MAF and related macro-forecasting blocks. |
| Bridge dispatch retirement | planned | Only after equivalence tests exist. |
| Simple/public sweeps | blocked | Wait for fixed full support and compiler guards. |
