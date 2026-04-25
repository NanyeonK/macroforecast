# Layer Contract Ledger

Date: 2026-04-25

This page is the canonical ledger for layer handoff, method-extension,
generator, payload, and artifact contracts. Other detail pages may explain why
a contract exists, but this page owns the status table and the producer /
consumer split.

Use this ledger when opening a new Layer 2 or Layer 3 runtime cell. A value
should not move from gated to operational until its contract has a producer, a
consumer, compiler validation, runtime validation, and at least one regression
test.

## Status Taxonomy

| Status | Meaning |
|---|---|
| `operational` | Runtime writes or consumes the contract in supported cells. |
| `operational_narrow` | Runtime works for a named slice, but broader use remains gated. |
| `gated_named` | Contract name and required semantics are known, but runtime is closed. |
| `future_design` | Design intent exists, but field schema is not stable enough for runtime gates. |
| `legacy_implicit` | Runtime behavior exists, but no exported contract constant/schema owns it yet. |

## Layer 3 View Taxonomy

| Layer 3 view | Meaning |
|---|---|
| `owned` | Layer 3 defines and produces this contract. Changes are Layer 3 work unless they require producer handoff changes. |
| `consumed` | Layer 3 consumes this contract but another layer owns its construction. Layer 3 can validate compatibility, not redefine it. |
| `future_dependency` | Layer 3 cannot open the related runtime cell until this contract exists or becomes operational. |
| `outside_layer3` | Contract belongs to another layer and is not a direct Layer 3 handoff. It may still affect compatibility through the final representation. |

## Core Layer Handoffs

| Contract | Owner | Layer 3 view | Producer | Consumer | Status | Notes |
|---|---|---|---|---|---|---|
| Layer 1 official frame handoff | Layer 1 | `outside_layer3` | raw/source adapters and official transform stage | Layer 2 representation builders | `legacy_implicit` | Covers official frame, target identity, horizons, information-set provenance, raw missing/outlier policy, and official transform/T-code reports. Should become an explicit schema before broader vintage/release-lag work. |
| `Layer2Representation` tabular handoff | Layer 2 | `consumed` | supported Layer 2 tabular builders | Layer 3 tabular generators | `operational` | Contains `Z_train`, `y_train`, `Z_pred`, feature names, block order/roles, fit state, alignment, leakage contract, and runtime provenance. Current Layer 3 capability matrix is built around this handoff. |
| `forecast_payload_v1` | Layer 3 | `owned` | scalar forecast generators | execution artifact writer and evaluation | `operational` | Public scalar payload with `y_pred`, `selected_lag`, `selected_bic`, `tuning_payload`, and `contract_version`. Legacy executor dictionaries are coerced into this shape. |
| `sequence_representation_contract_v1` | Layer 2 | `future_dependency` | future sequence/tensor representation builders | future Layer 3 sequence/tensor generators | `gated_named` | Required before sequence/tensor models enter full grids. Must define sample/origin axis, lookback axis, channel names, target/path alignment, fit state, leakage metadata, and missing/release-lag handling. |
| `exogenous_x_path_contract_v1` | Layer 1/2 boundary plus Layer 3 scenario setup | `future_dependency` | future scenario or future-X provider | raw-panel iterated forecast generators | `operational_narrow` | `hold_last_observed`, `observed_future_x`, and `scheduled_known_future_x` are operational explicit path kinds. `observed_future_x` is an oracle/ex-post path and must be marked in provenance. `scheduled_known_future_x` replaces only configured known-future predictor columns from future rows while holding other predictors at the origin row. Recursively forecast X, unavailable X, and broader vintage/release-lag variants remain gated until they have path-specific tests. |

## Future Contract Shape Requirements

The Layer 3 capability matrix exposes the following requirements as
machine-readable future-cell metadata (`schema_revision=6`). These are not
runtime permissions; they are the checklist for moving a gated cell to
operational.

### Sequence/Tensor Handoff

`sequence_representation_contract_v1` is produced by Layer 2 and consumed by
future sequence/tensor Layer 3 generators. It must define:

- `origin_index`;
- `sample_axis`;
- `lookback_axis`;
- `channel_names`;
- `target_alignment`;
- `fit_state`;
- `leakage_metadata`;
- `missing_release_lag_handling`.

Required gates before opening:

- sample/origin alignment test;
- lookback no-future-leakage test;
- channel-name schema test;
- generator payload-shape test.

`sequence_forecast_payload_v1` is produced by Layer 3 and consumed by the
artifact writer and evaluation layer. It must define:

- `origin_index`;
- `horizon`;
- `path_or_vector_payload`;
- `step_rows`;
- `aggregation_rule`;
- `payload_metrics`.

### Raw-Panel Iterated Forecasting

`exogenous_x_path_contract_v1` is the producer-side scenario contract consumed
by raw-panel iterated generators. It must define:

- `path_kind`;
- `origin_index`;
- `horizon_steps`;
- `predictor_names`;
- `x_path_frame_or_assumption`;
- `availability_mask`;
- `vintage_cutoff`;
- `release_lag_policy`;
- `no_lookahead_evidence`.

Allowed path kinds are:

- `observed_future_x`;
- `scheduled_known_future_x`;
- `hold_last_observed`;
- `recursive_x_model`;
- `unavailable`.

The first operational raw-panel iterated slices use
`path_kind='hold_last_observed'`, `path_kind='observed_future_x'`, and
`path_kind='scheduled_known_future_x'`. `hold_last_observed` is an explicit
deterministic scenario. `observed_future_x` is an oracle or ex-post analysis
path and must not be presented as real-time available information.
`scheduled_known_future_x` is a partial future-X path: configured known-future
predictor columns are read from future rows, while all other predictors stay at
their origin-available values. Later slices can add recursively forecast X.

`multi_step_raw_panel_payload_v1` is the Layer 3 artifact payload produced by
the iterated generator. It must define:

- `origin_index`;
- `horizon`;
- `step_predictions`;
- `final_horizon_prediction`;
- `target_history_updates`;
- `exogenous_x_path_ref`;
- `recursive_state_trace`;
- `payload_metrics`.

Required gates before opening:

- future-X availability test;
- release-lag mask test;
- origin-index alignment test;
- scenario-assumption manifest test;
- step-trace schema test;
- final-prediction projection test;
- recursive target-history test;
- JSONL schema test.

## Layer 2 Method-Extension Contracts

| Contract | Owner | Layer 3 view | Producer | Consumer | Status | Notes |
|---|---|---|---|---|---|---|
| `target_scale_contract_v1` | Layer 2 | `consumed` | target-scale compiler/runtime adapter | Layer 3 execution, inverse transform, evaluation | `operational` | Records target transform policy, normalization, evaluation scale, inversion support, and blockers. Interval/density wrappers currently require raw-level target scale, no target normalization, and no custom target transformer. |
| `custom_feature_block_callable_v1` | Layer 2 | `outside_layer3` | registered custom temporal, rotation, factor, or other feature block | Layer 2 block composer | `operational` | Block-local custom method contract. Requires train/pred feature frames, stable names, fit state, leakage metadata, and provenance. |
| `custom_feature_combiner_v1` | Layer 2 | `outside_layer3` | registered custom combiner | Layer 2 final `Z` composer | `operational` | Combines already-built named blocks into final `Z_train`/`Z_pred`. Must preserve names, block roles, row alignment, fit state, and leakage evidence. |
| `custom_final_z_selection_v1` | Layer 2 | `outside_layer3` | registered or built-in selector over final `Z` | Layer 2 final representation writer | `operational` | Records candidate, selected, and dropped feature names plus supervised/unsupervised selector fit state. Operational for supported registered custom block/combiner outputs. |
| `lag_polynomial_rotation_contract_v1` | Layer 2 | `outside_layer3` | MARX lag-polynomial rotation composer | Layer 2 raw-panel representation builder | `operational` | Defines predictor-major lag-polynomial feature names, runtime names, alignment, basis replacement, duplicate-base policy, and initial-lag fill policy. |
| `factor_score_history_contract_v1` | Layer 2 | `outside_layer3` | factor-to-rotation runtime fit state | Layer 2 factor-score rotation composers | `operational_narrow` | Operational for static-PCA `factor_then_marx` and MAF slices. Broader factor blocks need the same history/alignment evidence before opening. |
| `factor_score_rotation_contract_v1` | Layer 2 | `outside_layer3` | compiler metadata for MAF rotation | Layer 2/runtime provenance readers | `operational_narrow` | Companion metadata for factor-score moving-average rotation. It should stay aligned with `factor_score_history_contract_v1`; broader rotation contracts should avoid introducing another parallel name without a migration note. |
| `supervised_factor_block_contract_v1` | Layer 2 | `outside_layer3` | supervised factor block compiler/runtime | Layer 2 representation builder | `operational_narrow` | Used for supported supervised factor slices. Broader custom/supervised factor families still need explicit fit-state and leakage coverage. |
| Custom inverse transform contract | Layer 2/evaluation | `consumed` | future registered inverse callable | artifact writer and evaluation | `future_design` | Needed before custom inverse policies become operational. Must define model-scale payload input, original-scale output, failure behavior, and scale-specific metrics. |

## Layer 3 Generator Contracts

| Contract | Owner | Layer 3 view | Producer | Consumer | Status | Notes |
|---|---|---|---|---|---|---|
| `custom_model_v1` | Layer 3 | `owned` | registered custom model adapter | Layer 3 execution loop | `operational` | Custom models receive tabular Layer 2 representation context, training spec, seed context, and representation provenance. Operational for direct tabular paths, target-lag recursive paths, and the raw-panel iterated `hold_last_observed`, `observed_future_x`, and `scheduled_known_future_x` narrow slices. |
| Direct tabular generator protocol | Layer 3 | `owned` | tabular point/quantile generators | execution artifact writer | `legacy_implicit` | Operational for current scalar direct forecasts. Should become an explicit generator protocol if more generator families are added. |
| `path_average_target_protocol_v1` | Layer 2 | `consumed` | path-average target construction | Layer 3 path-average stepwise executor | `operational` | Layer 2 defines step targets and aggregation semantics; Layer 3 executes stepwise fits and writes path artifacts. |
| `path_average_stepwise_execution_v1` | Layer 3 | `owned` | path-average executor | artifact writer and evaluation | `operational` | Executes one supported scalar generator per step `1..h`, aggregates equal-weight path-average predictions, and writes `path_average_steps.csv`. |
| Sequence/tensor generator contract | Layer 3 | `owned` | future sequence/tensor executors | artifact writer and evaluation | `future_design` | Depends on `sequence_representation_contract_v1` and `sequence_forecast_payload_v1`. Must define accepted tensor family, training backend metadata, seed/early-stopping/convergence state, and payload shape. |
| Raw-panel iterated execution contract | Layer 3 | `owned` | raw-panel iterated generators | artifact writer and evaluation | `operational_narrow` | Operational for explicit `hold_last_observed`, oracle/ex-post `observed_future_x`, and configured `scheduled_known_future_x` paths plus fixed target-lag recursive target-history updates. Built-in scalar tabular generators and registered `custom_model_v1` adapters can consume these slices. Recursively forecast future-X path kinds remain gated. |

## Forecast Payload And Artifact Contracts

| Contract | Owner | Layer 3 view | Producer | Consumer | Status | Notes |
|---|---|---|---|---|---|---|
| `direction_forecast_payload_v1` | Layer 3 | `owned` | scalar payload wrapper | `predictions.csv`, `forecast_payloads.jsonl`, metrics | `operational` | Converts scalar forecasts into direction labels and probabilities relative to the current threshold and records direction hit metrics. |
| `interval_forecast_payload_v1` | Layer 3 | `owned` | scalar payload wrapper | `predictions.csv`, `forecast_payloads.jsonl`, metrics | `operational_narrow` | Builds symmetric Gaussian train-std intervals around scalar forecasts. This is a baseline interval wrapper, not a model-specific interval estimator. |
| `density_forecast_payload_v1` | Layer 3 | `owned` | scalar payload wrapper | `predictions.csv`, `forecast_payloads.jsonl`, metrics | `operational_narrow` | Builds Gaussian train-std density payloads and log-score columns around scalar forecasts. This is a baseline density wrapper, not a full distributional model. |
| `sequence_forecast_payload_v1` | Layer 3 | `owned` | future sequence/tensor generators | artifact writer and evaluation | `gated_named` | Must define path/vector payload shape, step-level versus horizon-level rows, metric aggregation, and JSONL schema. |
| `multi_step_raw_panel_payload_v1` | Layer 3 | `owned` | raw-panel iterated generators | artifact writer and evaluation | `operational_narrow` | Operational for `hold_last_observed`, `observed_future_x`, and `scheduled_known_future_x` raw-panel iterated point forecasts. It distinguishes step predictions, final horizon predictions, recursive target-history state, assumed, observed, or scheduled future-X path, configured scheduled-known columns, and path-level metrics. |
| Prediction row schema | Layer 5/evaluation boundary | `consumed` | execution artifact writer | evaluation, studies, downstream users | `legacy_implicit` | Existing `predictions.csv` is operational but not centrally versioned. Payload-family columns should not be added without updating this ledger or a future artifact schema page. |
| `forecast_payloads.jsonl` schema | Layer 5/evaluation boundary | `consumed` | execution artifact writer | downstream payload consumers | `operational` | Stores typed payload objects for scalar point, direction, interval, and density families. Future sequence/raw-panel payloads must extend this schema explicitly. |

## Validation Checklist

Before marking a contract `operational`, add or identify:

- compiler acceptance and rejection tests;
- runtime fit/apply or fit/predict tests;
- artifact schema assertions;
- manifest/provenance assertions;
- no-lookahead or alignment tests when the producer learns state;
- sweep invalid-cell behavior under `failure_policy=skip_failed_cell`;
- docs that state owner, producer, consumer, and current limits.

## Current Backlog

Highest-priority contract cleanup:

1. Make the Layer 1 official frame handoff explicit before deeper vintage,
   release-lag, or mixed-source work.
2. Decide whether the tabular `Layer2Representation` handoff needs an exported
   schema/version constant before public extension APIs expand further.
3. Define `sequence_representation_contract_v1` before opening sequence/tensor
   models in full grids.
4. Define `exogenous_x_path_contract_v1` and `multi_step_raw_panel_payload_v1`
   before opening raw-panel iterated forecasting.
5. Version the prediction row schema if payload families continue expanding.
