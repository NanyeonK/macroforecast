# Layer 3 Forecast Generator Boundary Audit

Date: 2026-04-24

Layer 3 is the forecast-generator layer. It consumes the Layer 2
representation handoff and produces forecasts. The Layer 0/1/2 migration makes
this boundary stricter: Layer 3 no longer owns feature representation choices,
target construction, official data treatment, or researcher preprocessing.

## Canonical Role

Layer 3 owns choices that determine how forecasts are generated after Layer 2
has produced `Z_train`, `y_train`, and `Z_pred`:

- model family and registered custom model;
- benchmark family and benchmark plugin execution;
- direct versus iterated forecast generation;
- forecast object, such as point mean, point median, or quantile;
- training window, minimum training size, training-start rule, refit policy,
  and lookback;
- model-order choices that are estimator behavior, such as AR BIC lag
  selection;
- validation split, hyperparameter search, tuning objective, and tuning budget;
- model seed, early stopping, convergence handling, cache, checkpointing, and
  execution backend.

Layer 3's canonical consumer contract is:

```text
fit_predict(forecast_generator, layer2_representation, training_spec) -> forecast_payload
```

The current tabular runtime handoff is `Layer2Representation`, with
`Z_train`, `y_train`, `Z_pred`, feature names, block order, block roles,
alignment, leakage contract, fit state, and runtime provenance.

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
forecast-generation protocol used to estimate and predict that target. If a
path-average target needs a multi-step execution protocol, the formula remains
Layer 2 and the multi-step fit/prediction rule is Layer 3.

### Target Scale Versus Forecast Object

Layer 2 owns target transforms, target normalization, target transformer
contracts, inverse-transform policy, and evaluation-scale provenance. Layer 3
owns what the model is asked to output: point mean, point median, quantile, or
future density/interval objects.

### Feature Selection Versus Hyperparameter Search

Layer 2 owns feature selection because it changes `Z`. Layer 3 owns
hyperparameter search because it chooses estimator settings conditional on the
given `Z`.

### Benchmarks

Benchmarks are Layer 3 because they generate forecasts. The official data frame
and feature representation are still supplied by Layers 1 and 2. Benchmark
scope and benchmark-window scoring details are evaluated later, but benchmark
forecast generation belongs here.

## Current Design State

The docs and runtime now mostly follow this split:

- Layer 2 owns `feature_builder`, `predictor_family`, `data_richness_mode`,
  `factor_count`, and all feature-block axes.
- Execution derives the runtime feature family from Layer 2 blocks first, with
  legacy `feature_builder` kept as fallback/provenance.
- Supported raw-panel and autoregressive tabular paths now use the
  `Layer2Representation` handoff.
- Registered custom Layer 3 models receive `custom_model_v1` context and Layer
  2 provenance.
- The compiler validates important Layer 2 x Layer 3 incompatibilities, such
  as raw-panel `Z` with `model_family='ar'`, raw-panel iterated forecasting,
  and quantile forecast objects with non-quantile models.

## Current Layer 3 Axes

| Group | Axes |
|---|---|
| Forecast generator | `model_family`, `benchmark_family`, `forecast_type`, `forecast_object`, `horizon_modelization` |
| Training window | `min_train_size`, `training_start_rule`, `outer_window`, `refit_policy`, `lookback` |
| Model order | `y_lag_count` for AR/model-order selection; fixed target-lag feature construction is Layer 2 provenance |
| Validation/search | `validation_size_rule`, `validation_location`, `embargo_gap`, `split_family`, `shuffle_rule`, `alignment_fairness`, `search_algorithm`, `tuning_objective`, `tuning_budget`, `hp_space_style` |
| Runtime discipline | `seed_policy`, `early_stopping`, `convergence_handling`, `logging_level`, `checkpointing`, `cache_policy`, `execution_backend` |

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
- `factor_ar_lags` still remains in `training_spec` as runtime compatibility
  debt because it conflates target-lag count and factor-lag count in old
  recipes. It should be split into Layer 2 target-lag / factor-lag metadata in
  a follow-up pass.
- `data_task_spec` still carries some migrated fields for compatibility, such
  as `forecast_type`, `forecast_object`, and
  `horizon_target_construction`. Compiler and docs should keep moving new
  generated recipes toward the canonical owners.
- `model_family` status is still value-level in the registry. The true
  capability is a matrix over `model_family`, Layer 2 feature runtime,
  `forecast_type`, and `forecast_object`.
- Built-in model executors still have separate autoreg/raw-panel wrappers.
  They should gradually converge toward the same
  `Layer2Representation -> forecast_payload` adapter shape used by custom
  models.

## Full Versus Simple

Full recipes may sweep Layer 2 representation axes and Layer 3 generator axes
together. The full route is the right place to expose representation x model
grids, invalid-cell pruning, and detailed per-cell provenance.

The simple API remains narrower:

- default single run;
- model comparison over `model_family`;
- fixed custom model;
- fixed custom preprocessor or target transformer where runtime support exists.

Simple representation/preprocessing sweeps should remain closed until full
route naming, result summaries, and invalid-cell reporting are stable.
