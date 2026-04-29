# Layer 3 Forecast Generator Boundary Audit

Date: 2026-04-24

Layer 3 is the forecast-generator layer. It consumes the Layer 2
representation handoff and produces forecasts. The Layer 0/1/2 migration makes
this boundary stricter: Layer 3 no longer owns feature representation choices,
target construction, FRED data treatment, or researcher preprocessing.

## Canonical Role

Layer 3 owns choices that determine how forecasts are generated after Layer 2
has produced `Z_train`, `y_train`, and `Z_pred`:

- forecast generator family, currently exposed through the compatibility axis
  `model_family`, and registered custom forecast generators;
- baseline assignment for forecast generators, currently exposed through the
  compatibility axis `benchmark_family`, plus benchmark plugin execution;
- direct versus iterated forecast generation;
- forecast object, such as point mean, point median, or quantile;
- training window, minimum training size, training-start rule, refit policy,
  and lookback;
- model-order choices that are estimator behavior, such as AR BIC lag
  selection;
- validation split, hyperparameter search, tuning objective, and tuning budget;
- estimator seed use, early stopping, and convergence handling.

Layer 3 does not own all runtime discipline. Runtime discipline is split:

| Discipline | Owner | Layer 3 role |
|---|---|---|
| Experiment execution control: failure policy, compute mode, reproducibility mode, broad cache/checkpoint policy | Layer 0 | follow the selected study policy |
| Data timing: vintage, release lag, contemporaneous information, and availability | Layer 1 | consume and validate the provided information set |
| Estimator training discipline: validation split, tuning, early stopping, convergence, model-specific seed use | Layer 3 | own and report estimator behavior |

Layer 3's canonical consumer contract is:

```text
fit_predict(forecast_generator, layer2_representation, training_spec) -> forecast_payload
```

Named contract statuses are centralized in `layer_contract_ledger.md`. This
audit uses those names only to explain the Layer 3 boundary.

The scalar runtime forecast payload contract is `forecast_payload_v1`:

| Field | Meaning |
|---|---|
| `y_pred` | Scalar forecast on the model target scale. Inversion and metric-scale conversion happen after the payload is returned. |
| `selected_lag` | Estimator-side lag/order selected or fixed by the forecast generator. Fixed target-lag columns already present in `Z` remain Layer 2 provenance. |
| `selected_bic` | BIC value when BIC order selection is used; `NaN` for fixed-order or non-BIC generators. |
| `tuning_payload` | Optional estimator/runtime metadata. Runtime coercion adds `forecast_payload_contract=forecast_payload_v1`. |

The current tabular runtime handoff is `Layer2Representation`, with
`Z_train`, `y_train`, `Z_pred`, feature names, block order, block roles,
alignment, leakage contract, fit state, and runtime provenance.

Layer 3 also supports typed payload-family wrappers around the scalar
generator:

| Forecast object | Payload contract | Current runtime |
|---|---|---|
| `direction` | `direction_forecast_payload_v1` | Converts scalar forecasts into up/down payloads relative to a zero threshold and records hit indicators. |
| `interval` | `interval_forecast_payload_v1` | Builds a symmetric Gaussian interval around the scalar forecast using train-window target dispersion. |
| `density` | `density_forecast_payload_v1` | Builds a Gaussian density payload around the scalar forecast using train-window target dispersion and records log scores. |

These wrappers make payload type, artifact columns, and manifest contracts
explicit without pretending that they are model-specific distributional
estimators. Sequence/tensor handoff and raw-panel iterated forecasting remain
gated until their producer handoff and scenario contracts exist.
Path-average point forecast execution is operational through the stepwise
Layer 3 protocol.

## Non-Ownership

Layer 3 must not decide how the data or representation was built. These are
not Layer 3 responsibilities:

- dataset/source/frequency/information-set decisions;
- official transform policy, release-lag policy, raw-source missing/outlier
  policy, or variable-universe selection;
- horizon target construction formulas, such as future level, growth, or
  path-average target formulas;
- target/X post-frame transforms, missing imputation, outlier handling,
  scaling, normalization, smoothing, feature selection, or dimensionality
  reduction;
- target-lag, X-lag, factor, level, temporal, rotation, deterministic, or
  custom feature-block construction;
- factor count when it is a representation dimension;
- final `Z` block combination.

Layer 3 may reject a selected representation when the chosen forecast generator
cannot consume it. That is compatibility validation, not ownership of the
representation.

## Boundary Splits

### Target Lags Versus AR Order

`target_lag_block=fixed_target_lags` is Layer 2 feature construction. It means
the final `Z` contains target-history columns. A direct ridge, lasso, or custom
model can consume those columns without becoming an AR-BIC model.

AR BIC lag selection is Layer 3 estimator behavior. It chooses model order
inside the autoregressive forecast generator.

### Target Construction Versus Forecast Generation

Layer 2 owns the supervised target representation, including level, difference,
growth, log growth, and path-average target formulas. Layer 3 owns the
forecast-generation protocol used to estimate and predict that target. For
path-average targets, the formula remains Layer 2 and the multi-step
fit/predict/aggregate rule is Layer 3.

### Target Scale Versus Forecast Object

Layer 2 owns target transforms, target normalization, target transformer
contracts, inverse-transform policy, and evaluation-scale provenance. Layer 3
owns what the model is asked to output: point mean, point median, quantile, or
future density/interval objects.

### Feature Selection Versus Hyperparameter Search

Layer 2 owns feature selection because it changes `Z`. Layer 3 owns
hyperparameter search because it chooses estimator settings conditional on the
given `Z`.

### Forecast Generators And Benchmarks

Benchmarks are Layer 3 because they generate forecasts, but they should not be
treated as a separate species from models. The canonical concept is one
forecast-generator registry with comparison roles:

| Concept | Current compatibility axis | Canonical meaning |
|---|---|---|
| Candidate generator | `model_family` | Forecast generator being evaluated as the candidate method. |
| Baseline generator | `benchmark_family` | Forecast generator assigned the benchmark/baseline role. |

An AR generator, historical mean, random walk, ridge model, or registered
custom generator can be a candidate or a baseline depending on the comparison
design. The current `model_family` and `benchmark_family` names remain accepted
for recipe compatibility, but docs should describe them as generator family and
generator role assignment. The FRED data frame and feature representation
are still supplied by Layers 1 and 2.

## Current Design State

The docs and runtime now mostly follow this split:

- Layer 2 owns `feature_builder`, `predictor_family`, `data_richness_mode`,
  `factor_count`, and all feature-block axes.
- Execution derives the runtime feature family from Layer 2 blocks first, with
  legacy `feature_builder` kept as fallback/provenance.
- Supported raw-panel and autoregressive tabular paths now use the
  `Layer2Representation` handoff.
- Built-in raw-panel factor-model adapters (`pcr`, `pls`, and
  `factor_augmented_linear`) also consume the same `Layer2Representation`
  bundle instead of rebuilding predictor frames beside Layer 2.
- Factor-to-rotation choices such as `factor_rotation_order=factor_then_rotation`,
  `factor_then_marx`, and MAF factor-score rotation are Layer 2 representation
  decisions. Layer 3 receives only the final `Z` bundle, fit-state provenance,
  and feature names.
- Path-average target constructions are operational for point forecasts: Layer
  3 fits one supported generator per step, aggregates with equal weights,
  writes `path_average_steps.csv`, and records the aggregate row in
  `predictions.csv`. The current runtime requires raw-level target scale, no
  target normalization, and no custom target transformer.
- Direction, interval, and density forecast objects are operational as typed
  payload-family wrappers over supported scalar point generators. Runtime
  writes payload columns into `predictions.csv`, writes
  `forecast_payloads.jsonl`, and records the active payload contract in the
  manifest. Interval/density wrappers currently require raw-level target scale,
  no target normalization, and no custom target transformer.
- Registered custom Layer 3 models receive `custom_model_v1` context and Layer
  2 provenance.
- FRED-SD mixed-frequency direct generators now include `midas_almon`, generic
  `midasr` with `midasr_weight_family` values `nealmon`, `almonp`, `nbeta`,
  `genexp`, and `harstep`, and the legacy `midasr_nealmon` alias. They consume
  Layer 2 native-frequency block payloads; they do not own FRED-SD
  panel-shaping decisions.
- State-space nowcasting remains explicitly gated. The reserved
  `state_space_mixed_frequency_payload_v1` contract must define observation
  equations, missing-observation handling, filter/smoother state payloads, and
  forecast-origin update semantics before any `state_space_run` cell becomes
  executable.
- The compiler validates important Layer 2 x Layer 3 incompatibilities, such
  as raw-panel `Z` with `model_family='ar'`, raw-panel iterated forecasting,
  and quantile forecast objects with non-quantile models.
- New compiled manifests include `layer3_capability_matrix`, which records the
  active compatibility cell
  `model_family x feature_runtime x forecast_type x forecast_object`. The
  canonical interpretation is
  `forecast_generator_family x representation_runtime x forecast_protocol x forecast_object`.
  The old names remain manifest compatibility names; the matrix also records
  `canonical_dimensions`, `dimension_aliases`, and `canonical_active_cell`.
- The same matrix includes a status catalog, active payload contract names, and
  reserved future cells for sequence/tensor runtimes and raw-panel iterated
  forecasting. The canonical status, producer, consumer, and validation backlog
  for these contracts lives in `layer_contract_ledger.md`.

## Current Layer 3 Axes

| Group | Axes |
|---|---|
| Forecast generator | `model_family` as compatibility alias for `forecast_generator_family`; `benchmark_family` as compatibility alias for baseline generator role; `forecast_type`, `forecast_object`, `horizon_modelization` |
| Generator subfamily | `midasr_weight_family` for the restricted MIDAS weight family under `model_family=midasr`; `nealmon`, `almonp`, `nbeta`, `genexp`, and `harstep` are operational-narrow. `harstep` requires `midas_max_lag=20`; compiler defaults to 20 when that family is selected and no explicit lag is supplied. |
| Training window | `min_train_size`, `training_start_rule`, `outer_window`, `refit_policy`, `lookback` |
| Model order | `y_lag_count` for AR/model-order selection; fixed target-lag feature construction is Layer 2 provenance |
| Validation/search | `validation_size_rule`, `validation_location`, `embargo_gap`, `split_family`, `shuffle_rule`, `alignment_fairness`, `search_algorithm`, `tuning_objective`, `tuning_budget`, `hp_space_style` |
| Estimator training discipline | `early_stopping`, `convergence_handling`, model-specific seed use constrained by Layer 0 `reproducibility_mode` |
| Execution/runtime control consumed from Layer 0 | `seed_policy`, `logging_level`, `checkpointing`, `cache_policy`, `execution_backend` when these are broad run-control choices rather than estimator-specific choices |

## Compatibility Debt

The boundary is defined, but these cleanup items remain:

- The first `training_spec` cleanup pass moved `data_richness_mode`,
  `target_lag_selection`, `target_lag_count`, `custom_preprocessor`, and
  `target_transformer` out of newly generated `training_spec` and into
  `layer2_representation_spec`. Runtime readers for custom preprocessors and
  target transformers read Layer 2 metadata first, falling back to legacy
  `training_spec` only for old recipes.
- The second cleanup pass moved `factor_count`, `fixed_factor_count`, and
  `max_factors` out of newly generated `training_spec`. Factor runtimes now
  read Layer 2 factor-block metadata first and fall back to legacy
  `training_spec` only for old recipes.
- The third cleanup pass split legacy `factor_ar_lags` into Layer 2
  `target_lag_count` and factor-block `factor_lag_count` metadata. Runtime
  readers still accept old `training_spec.factor_ar_lags` as fallback only.
- New compiled specs now emit Layer 3 forecast/window fields in
  `training_spec`: `forecast_type`, `forecast_object`, `min_train_size`,
  `training_start_rule`, and `training_start_date`. Runtime readers still
  fall back to old `data_task_spec` locations for compatibility.
- New compiled specs now emit the main Layer 2 target/input/deterministic
  representation fields in `layer2_representation_spec`:
  `horizon_target_construction`, `predictor_family`,
  `contemporaneous_x_rule`, `deterministic_components`, and
  `structural_break_segmentation`. Runtime readers still fall back to old
  `data_task_spec` locations for compatibility.
- `model_family` is still the public compatibility axis for candidate forecast
  generators, but the canonical concept is `forecast_generator_family`.
  Runtime support is represented in `layer3_capability_matrix`. The current
  matrix covers the operational tabular cells: target-lag-only iterated point
  generators, raw-panel direct point generators, quantile-linear
  median/quantile outputs, and direction/interval/density payload-family
  wrappers over scalar point generators.
- `benchmark_family` is still the public compatibility axis for baseline
  generators. Canonically, this is forecast-generator role assignment rather
  than a separate Layer 3 family. The current runtime still has a separate
  benchmark executor path for compatibility.
- Built-in model executors still have separate autoreg/raw-panel wrappers, but
  supported sklearn, custom, and factor-model paths now attach
  `Layer2Representation` metadata to their tuning payloads. Sequence/tensor
  adapters remain outside this tabular handoff.

## Full Versus Simple

Full recipes may sweep Layer 2 representation axes and Layer 3 generator axes
together. The full route is the right place to expose representation x model
grids, invalid-cell pruning, and detailed per-cell provenance.

The simple API remains narrower:

- default single run;
- forecast-generator comparison over the current `model_family` compatibility
  axis;
- fixed custom model;
- fixed custom preprocessor or target transformer where runtime support exists.

Simple representation/preprocessing sweeps should remain closed until full
route naming, result summaries, and invalid-cell reporting are stable.
