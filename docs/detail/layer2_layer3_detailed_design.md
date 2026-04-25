# Layer 2 And Layer 3 Detailed Design

Date: 2026-04-25

This document fixes the detailed design for the next Layer 2 and Layer 3
extension work after the Layer 0, Layer 1, and Layer 2 migration cleanup.

The current migration target is closed for supported tabular full recipes:
Layer 2 builds a canonical `Layer2Representation`, Layer 3 consumes that
representation, and broad Layer 2 x Layer 3 grids report runnable and invalid
cells explicitly. The remaining work is not boundary cleanup. It is research
method support: new representation composers, new forecast-generation
protocols, and public sweep surfaces.

Named papers should remain reference presets over general primitives. A paper
can motivate a preset, but the package language should expose reusable Layer 2
and Layer 3 choices.

## Design Goals

1. Let method researchers add or sweep representation methods in Layer 2.
2. Let method researchers add or sweep forecast generators in Layer 3.
3. Keep fair comparison controls shared across built-in and custom methods:
   Layer 0 task, Layer 1 data treatment, split, horizon, benchmark, and
   evaluation.
4. Keep all learned representation state inside the relevant training window.
5. Treat invalid Layer 2 x Layer 3 cells as auditable compiler/runtime results,
   not silent drops.
6. Keep simple APIs conservative until full recipes have stable naming,
   provenance, and result summaries.

## Boundary

Layer 2 owns the supervised representation:

- target construction and target-side scale;
- target-history feature blocks;
- predictor lag blocks;
- factor blocks;
- level add-back blocks;
- temporal feature blocks;
- lag-polynomial and moving-average rotation blocks;
- block-local custom feature methods;
- final block combination into `Z`;
- final representation-level preprocessing and feature selection;
- representation fit state, leakage metadata, and feature provenance.

Layer 3 owns forecast generation after Layer 2 has produced the representation:

- model family and registered custom model;
- benchmark family;
- direct, iterated, and future multi-step execution protocols;
- forecast object, such as point, quantile, interval, or density;
- training window, refit policy, validation, tuning, and search;
- estimator-side order selection;
- execution backend, checkpointing, convergence handling, and cache;
- forecast payload emission.

Layer 3 may reject a representation it cannot consume. That is compatibility
validation, not Layer 3 ownership of feature construction.

## Current Closed Contract

The current runtime contract is tabular:

```text
Layer 1 official frame -> Layer 2 Layer2Representation -> Layer 3 ForecastPayload
```

`layer_contract_ledger.md` is the canonical status ledger for named contracts.
This design page explains why each contract exists and how phases compose; it
should not duplicate the full producer/consumer/status table.

Layer 2 emits `Layer2Representation` with:

| Field | Meaning |
|---|---|
| `Z_train` | Training feature matrix for the current origin/window. |
| `y_train` | Training target vector aligned to `Z_train`. |
| `Z_pred` | One-row prediction feature matrix. |
| `feature_names` | Public feature names in matrix order. |
| `block_order` | Ordered feature blocks used to form `Z`. |
| `block_roles` | Feature name to block-role mapping. |
| `fit_state` | Per-window state for learned representation steps. |
| `alignment` | Row timing and origin alignment metadata. |
| `leakage_contract` | Leakage rule, currently forecast-origin-only. |
| runtime provenance | Feature builder, runtime builder, bridge source, and compatibility aliases. |

Layer 3 returns `ForecastPayload` v1 with:

| Field | Meaning |
|---|---|
| `y_pred` | Scalar prediction on the model target scale. |
| `selected_lag` | Estimator-side selected/fixed order when relevant. |
| `selected_bic` | BIC value for BIC-order paths; `NaN` otherwise. |
| `tuning_payload` | Estimator/runtime metadata. |
| `contract_version` | `forecast_payload_v1`. |

This contract is enough for current point and quantile tabular paths. It is not
enough for sequence/tensor models, path-average multi-step execution, interval
payloads, density payloads, or raw-panel iterated forecasting.

## Layer 2 Detailed Design

### Representation Families

Layer 2 needs two representation families:

| Family | Status | Consumer |
|---|---|---|
| `tabular_z` | Operational | Current built-in, sklearn, custom, factor, AR-compatible tabular paths. |
| `sequence_tensor` | Future | Sequence/tensor forecast generators and deep temporal models. |

`Layer2Representation` remains the public tabular handoff. Sequence/tensor
support should not overload `Z_train` with ambiguous shapes. It needs a
separate handoff contract before sequence/tensor models enter full grids.

Proposed future `Layer2SequenceRepresentation` fields:

| Field | Meaning |
|---|---|
| `X_sequence_train` | `n_train x lookback x n_features` tensor or equivalent array. |
| `y_train` | Target vector or target path aligned to sequence rows. |
| `X_sequence_pred` | One prediction-origin sequence. |
| `feature_names` | Names for sequence channels. |
| `lookback` | Sequence length and alignment rule. |
| `row_index` | Train-row origin dates. |
| `channel_roles` | Channel to block-role mapping. |
| `fit_state` | Sequence scaler/encoder/window state. |
| `alignment` | Origin, lookback, horizon, and target timing metadata. |
| `leakage_contract` | No-lookahead evidence. |

Sequence/tensor support is a separate representation family, not a Layer 3
model switch over the current `Layer2Representation`.

### Custom Feature Blocks

Current custom feature blocks are block-local Layer 2 methods. They are the
right hook when a researcher adds one new source of features: custom temporal
summaries, custom rotations, custom factors, or other block-level transforms.

Contract: `custom_feature_block_callable_v1`.

Required output:

| Field | Meaning |
|---|---|
| `train_features` | Training feature frame aligned to the current window. |
| `pred_features` | One-row prediction feature frame. |
| `feature_names` | Stable public names. |
| `runtime_feature_names` | Matrix column names if different from public names. |
| `fit_state` | Fitted parameters or selected source variables. |
| `leakage_metadata` | At least `lookahead=forbidden` or an explicit exception. |
| `provenance` | Method name, options, source columns, and sweep-relevant metadata. |

This is already the right unit for adding individual Layer 2 methods.

### Custom Combiner

`feature_block_combination=custom_combiner` is the broader Layer 2 extension.
It is not a block. It owns how already-built blocks are combined into final
`Z`.

Operational contract: `custom_feature_combiner_v1`.

Inputs:

| Input | Meaning |
|---|---|
| `blocks_train` | Ordered map from block name to train feature frame. |
| `blocks_pred` | Ordered map from block name to prediction feature frame. |
| `y_train` | Read-only aligned target vector. |
| `origin` | Forecast origin. |
| `horizon` | Forecast horizon. |
| `block_metadata` | Source block roles, fit state, and leakage metadata. |
| `context` | Recipe axes, split/window metadata, target spec, and seed context. |

Outputs:

| Output | Meaning |
|---|---|
| `Z_train` | Final training matrix. |
| `Z_pred` | Final prediction row. |
| `feature_names` | Public final feature names. |
| `block_roles` | Feature name to role mapping. |
| `fit_state` | Combiner-level fit state. |
| `leakage_metadata` | Combiner no-lookahead evidence. |
| `provenance` | Combiner method, options, source blocks, and selected interactions. |

Validation rules:

- `Z_train` and `Z_pred` must have identical columns and numeric dtypes.
- Output row count must match `y_train`.
- Feature names must be unique after namespacing.
- The combiner may read `y_train` for supervised representation learning, but
  only inside the current training window.
- The combiner must not mutate input block frames.
- The final `Layer2Representation.block_order` should include both source
  blocks and a combiner marker.

This contract enables arbitrary research representation composition without
placing feature construction inside Layer 3 models.

### Final-Z Selection After Custom Blocks

Current built-in final-`Z` selection is open for the supported static-factor
slice and for registered custom block/combiner outputs.

Operational contract: `custom_final_z_selection_v1`.

Required behavior:

- selection operates after block composition;
- selector receives final `Z_train`, `Z_pred`, `y_train`, feature names,
  block roles, and provenance;
- selector returns selected train/pred matrices, selected feature names,
  dropped feature names, and selector fit state;
- selector records whether it is supervised or unsupervised;
- supervised selectors fit only inside the current training window;
- selected/dropped custom feature names remain traceable to the source custom
  block or custom combiner.

Use `feature_selection_semantics=select_after_custom_blocks` when custom
representation columns should participate in the same selection sweeps as
built-ins.

### Factor-To-Rotation Composers

The current runtime supports `marx_rotation` on raw-panel X,
`marx_then_factor`, `factor_then_marx`, and `maf_rotation` with static PCA
factors. The factor-to-rotation modes require factor-score history, not just
one prediction-row factor score.

Required factor-score history contract:

| Field | Meaning |
|---|---|
| `factor_scores_train` | Factor score time series for all train rows. |
| `factor_score_pred_base` | Forecast-origin factor score if needed. |
| `factor_names` | Stable factor names. |
| `factor_loadings` | Loadings or transformation metadata. |
| `score_index` | Dates/origins associated with score rows. |
| `source_panel` | Source block used before factor extraction. |
| `fit_window` | Recursive training window metadata. |

`factor_then_marx` design:

- fit factors on the X-side source block inside the current training window;
- build lag-polynomial rotations on factor-score histories;
- produce names such as `factor_1_marx_ma_lag1_to_lag3`;
- preserve predictor-source loadings in factor fit state;
- record that the lag-polynomial source is factor scores, not raw predictors.

`maf_rotation` design:

- fit the source factor block inside the current training window;
- rotate factors into moving-average factors or another explicitly named
  factor-to-temporal basis;
- return rotation loadings, score alignment, and no-lookahead metadata;
- keep MAF as a general factor-to-rotation primitive, not a paper-specific
  layer name.

Both composers are operational for `pca_static_factors`. Alignment tests must
remain part of the acceptance suite before additional factor blocks or custom
factor-score rotations are opened.

### Target-Side Custom Inverse

`inverse_transform_policy=custom` is a Layer 2 target-scale item. It changes
how model-scale predictions are mapped to evaluation scale after Layer 3
returns a payload.

Proposed contract: `custom_inverse_transform_v1`.

Inputs:

- model-scale forecast payload;
- target transformer fit state;
- raw-level target history available at the forecast origin;
- horizon and target-construction metadata;
- evaluation-scale request.

Outputs:

- transformed-scale forecast value/object;
- original-scale forecast value/object when requested;
- inversion diagnostics;
- failure mode when inversion is undefined.

This contract should remain outside Layer 3. Layer 3 predicts the target it was
given; Layer 2/evaluation converts scale.

## Layer 3 Detailed Design

### Forecast Payload Families

`ForecastPayload` v1 is scalar and point-oriented. Additional forecast objects
use typed payload-family wrappers rather than overloading `y_pred`.

Current payload family:

| Payload | Status | Fields |
|---|---|---|
| `PointForecastPayload` | Operational via `ForecastPayload` v1 | point prediction plus estimator metadata. |
| `QuantileForecastPayload` | Partly operational through current quantile paths | quantile level(s), quantile predictions, calibration metadata. |
| `DirectionForecastPayload` | Operational | direction label, up probability, threshold metadata, and hit indicators. |
| `IntervalForecastPayload` | Operational as scalar wrapper | lower/upper bounds, coverage level, method metadata, and coverage indicators. |
| `DensityForecastPayload` | Operational as scalar wrapper | Gaussian mean/variance metadata and log-score columns. |

The current direction/interval/density wrappers consume the existing scalar
generator and write explicit payload columns plus `forecast_payloads.jsonl`.
They are baseline payload contracts, not model-specific distributional
estimators. Sequence/tensor and raw-panel iterated payloads remain gated until
their producer handoff and scenario contracts exist. The canonical payload
contract status table lives in `layer_contract_ledger.md`.

### Path-Average Execution

Layer 2 records `path_average_target_protocol_v1`. Layer 3 executes that
protocol for point forecasts through `path_average_stepwise_execution_v1`.

Layer 2 responsibilities:

- define the step target specs for steps `1..h`;
- define the aggregation formula;
- expose scale and inversion metadata for each step;
- validate that the target construction has a path-average protocol.

Layer 3 responsibilities:

- for each origin and each step, build the step-specific Layer 2 target;
- fit/predict the selected generator for each step;
- collect one forecast payload per step;
- aggregate step predictions into the horizon-level forecast object;
- write per-step and aggregate artifacts.

Current artifacts:

| Artifact | Meaning |
|---|---|
| `path_average_steps.csv` | One row per origin, step, model, and payload value. |
| `predictions.csv` | Horizon-level aggregate prediction rows with `path_average_runtime`, step count, aggregation rule, construction-scale metrics, and level-scale provenance. |
| `manifest.json` | `path_average_step_rows`, optional `path_average_steps_file`, target protocol metadata, payload contract, and standard failure/provenance fields. |
| `metrics.json` | Standard metrics computed on the aggregate path-average target scale plus original-level scale metrics. |

Current point-forecast support covers autoregressive target-lag generators and
supported tabular/raw-panel generators. It currently requires deterministic
target scale contracts: `target_transform_policy='raw_level'`,
`target_normalization='none'`, and no custom target transformer.

### Raw-Panel Iterated Forecasting

Raw-panel iterated forecasting is a Layer 3 execution protocol, but it needs
more than the current tabular `Z_pred`.

Operational narrow slice:

- `exogenous_x_path_contract_v1.path_kind` in `{'hold_last_observed',
  'observed_future_x', 'scheduled_known_future_x', 'recursive_x_model'}`;
- `target_lag_block='fixed_target_lags'`;
- `forecast_object='point_mean'`;
- built-in scalar tabular model generators or registered custom models using
  `custom_model_v1`;
- raw-level target and X, no target normalization, no custom target
  transformer, and no extra Layer 2 preprocessing.

Still-gated dependency:

- `exogenous_x_path_contract_v1`;
- `multi_step_raw_panel_payload_v1`;
- richer recursively forecast future-X model families beyond `ar1`;
- non-point forecast payloads;
- transformed/normalized target scale composition.

The operational path kinds are
`exogenous_x_path_contract_v1.path_kind='hold_last_observed'`,
`path_kind='observed_future_x'`, and
`path_kind='scheduled_known_future_x'`, plus
`path_kind='recursive_x_model'` when
`recursive_x_model_family='ar1'`. `hold_last_observed` is explicitly a
scenario assumption, not knowledge of future X. `observed_future_x` is an
oracle or ex-post path and must be marked as future information.
`scheduled_known_future_x` is a partial future-X path: only configured
known-future predictor columns use future rows, while all other predictors stay
at the origin row. The first recursive-X model slice fits a per-predictor AR(1)
on origin-available predictor history and recursively generates later-step X
without consuming observed future X values. These paths should write a step
trace, the future-X path reference, recursive target-history updates,
configured scheduled-known columns or recursive-X model family when
applicable, and a final horizon prediction under
`multi_step_raw_panel_payload_v1`.

Broader raw-panel iterated cells remain blocked by the Layer 3 capability
matrix until richer recursively forecast future-X model families have explicit
availability, release-lag, origin-alignment, and artifact tests.

### Sequence/Tensor Forecast Generators

Sequence/tensor models belong in Layer 3 only after Layer 2 has emitted a
sequence/tensor representation family.

Layer 3 responsibilities:

- consume `Layer2SequenceRepresentation`;
- fit sequence/tensor estimators;
- return the appropriate forecast payload family;
- record training backend, seed, early stopping, and convergence metadata.

Layer 2 responsibilities:

- define windows, channels, lookback, scaling, and leakage metadata;
- keep target construction and sequence alignment out of the model closure.

Sequence/tensor runtime should be a separate capability matrix feature runtime,
not a hidden extension of the current raw-panel tabular cell.

## Sweep Semantics

Full recipes may request broad Cartesian sweeps over Layer 2 and Layer 3 axes.
The compiler should materialize cells first, then validate each concrete cell.

Status taxonomy:

| Status | Meaning |
|---|---|
| `operational` | Cell compiles and can execute in the current runtime. |
| `blocked_by_incompatibility` | Values are valid separately but cannot compose. |
| `not_supported` | Contract is not implemented yet. |
| `registry_only` | Name is reserved but not executable. |
| `future_contract` | Design is documented, registry/runtime still closed. |

The study manifest should keep:

- axis values for every materialized cell;
- compiler status and blocked reasons;
- Layer 3 capability cell;
- runtime status;
- metrics when execution succeeded;
- skipped/failed counts.

This lets researchers sweep broadly while preserving auditability.

## Full Versus Simple

Full route:

- exposes Layer 2 x Layer 3 grids;
- reports invalid cells;
- writes per-cell manifests;
- supports custom blocks, custom combiners, and custom models when contracts
  are registered;
- is the first place to open new research representation choices.

Simple route:

- keeps stable defaults;
- may compare model families;
- may use a fixed registered custom model or fixed registered custom
  preprocessor;
- should not expose broad representation sweeps until full result summaries
  and invalid-cell reporting are stable.

## Recommended Implementation Order

### Phase 1: Custom Combiner And Custom Final-Z Selection

Status: operational for registered custom combiners and operational built-in
feature-selection policies over registered custom block/combiner outputs.

Reason: this directly supports method researchers who want to add a Layer 2
representation and compare it with built-ins under the same Layer 3 models.
It reuses the current tabular `Layer2Representation`.

Acceptance:

- registered custom combiner compiles in full recipes;
- combiner receives named block frames and returns final `Z`;
- custom output names appear in manifests and model context;
- final-`Z` selection can select/deselect custom columns;
- invalid/missing callable cells are skipped or blocked with precise reasons.

### Phase 2: Path-Average Layer 3 Execution

Reason: Layer 2 protocol metadata already exists; missing work is clearly
Layer 3 scheduling and artifact emission.

Acceptance:

- path-average target constructions execute for supported tabular generators;
- per-step payloads and aggregate payloads are recorded;
- metrics are emitted on requested scales;
- compiler registry marks path-average constructions operational;
- failed step cells are auditable.

### Phase 3: Factor-To-Rotation Composers

Reason: `factor_then_marx` and MAF need factor-score history and alignment
tests. They are Layer 2 representation composers; Layer 3 consumes their final
`Z` bundle like any other tabular representation.

Acceptance:

- factor-score history contract exists;
- `factor_then_marx` produces stable names and provenance;
- MAF rotation records score/loadings/alignment metadata;
- no-lookahead tests cover train and prediction rows.

### Phase 4: Custom Inverse Transform

Reason: target-scale customization affects forecast artifacts and evaluation.
It should land after payload contracts are stable enough for scale conversion.

Acceptance:

- registered inverse callable converts model-scale payloads;
- original/transformed-scale artifacts are consistent;
- undefined inverse behavior is explicit and test-covered.

### Phase 5: Forecast Payload Family

Status: direction, interval, and density are now operational as typed wrappers
over scalar point generators. Do not overload scalar `y_pred`.

Acceptance:

- payload classes or tagged payload schemas exist;
- Layer 3 capability matrix records payload contract per active cell;
- artifacts preserve payload type, scale, and scoring metadata;
- `layer_contract_ledger.md` records owner, producer, consumer, status,
  validators, and remaining gates for each payload contract.

The current `layer3_capability_matrix_v1` schema revision opens
`direction_forecast_payload_v1`, `interval_forecast_payload_v1`, and
`density_forecast_payload_v1`. Sequence/tensor and raw-panel iterated payloads
remain gated in the ledger until their handoff/scenario contracts are defined.

### Phase 6: Sequence/Tensor Representation Handoff

Reason: this opens a new representation family. It should not be mixed into
current tabular `Z` runtime by convention.

Acceptance:

- `Layer2SequenceRepresentation` or equivalent contract exists;
- sequence/tensor Layer 3 executors consume only that contract;
- full sweeps can report tabular-versus-sequence incompatibilities.

### Phase 7: Public Simple API Exposure

Reason: public convenience should follow the stable full route.

Acceptance:

- simple API exposes only stable presets;
- no registry-only or future-contract values leak into simple defaults;
- documentation maps simple presets to full recipe axes.

## Test And Documentation Requirements

Every implementation phase needs:

- compile tests for valid and invalid cells;
- manifest/provenance tests;
- leakage/alignment tests for learned Layer 2 behavior;
- sweep tests with `failure_policy=skip_failed_cell`;
- docs build;
- at least one full recipe example when the surface becomes user-facing.

The first test for any new contract should prove that an unregistered or
unsupported value is blocked with a precise reason. Runtime success tests come
after that gate is explicit.
