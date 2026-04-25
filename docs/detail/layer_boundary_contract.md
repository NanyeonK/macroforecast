# Layer Boundary Contract

Date: 2026-04-22

This document is the canonical boundary contract for the recipe layers after the
Layer 1/2 cleanup decision.

For named runtime and extension schemas, use
`layer_contract_ledger.md` as the canonical status ledger. This page owns the
layer-role split; the ledger owns contract status, producer, consumer,
validation, and backlog.

## Layer 0: Study Design

Owns study grammar and execution shape:

- `research_design`
- `experiment_unit`
- `axis_type`
- `failure_policy`
- `reproducibility_mode`
- `compute_mode`

Layer 0 must not know dataset semantics, preprocessing semantics, model
families, or metrics.

## Layer 1: Official Data Frame

Owns the official data frame before researcher-specific transformations:

- dataset identity
- source adapter / loader
- frequency
- information set and release-lag availability
- target identity, horizons, sample start/end
- official dataset transformation policy and target/X transform scope
- official availability handling
- raw-source missing/outlier policy before official transforms/T-codes
  (`raw_missing_policy`, `raw_outlier_policy`)
- raw eligible variable universe
- contemporaneous information-set rule

Layer 1 output is an official frame plus provenance reports. It should be enough
to reproduce "what data were available to the study" before model-specific
choices.

In full mode, Layer 1 may clean or flag raw-source missing values and
raw-source outliers before official transforms/T-codes. That order must be
recorded because it differs from imputing or clipping after transformed model
inputs already exist.

## Layer 2: Research Preprocessing / Feature Representation

Owns transformations and feature representations researchers can vary within
the same official data frame:

- target transforms beyond official dataset codes
- X transforms beyond official dataset codes
- scaling and normalization
- post-transform/model-input missing imputation algorithms
- post-transform/model-input outlier handling
- smoothing / filters
- PCA, static factors, dimensionality reduction
- feature selection
- predictor family and feature-block selection
- feature builders that construct the model input matrix `Z`
- factor count and other representation dimensions
- deterministic features, including trends, seasonals, and break dummies
- custom preprocessors
- fit scope and leakage discipline
- inverse transform and evaluation scale

Layer 2 can handle values that originated as raw-source missing/outliers, but
only after Layer 1 has produced the selected official or raw feature frame. In
that case the treatment may be mixed with transform-induced missing values and
other preprocessing artifacts; full provenance should preserve that ordering.

Layer 2's canonical output is `Z_train`/`Z_pred`, plus feature names, block
metadata, and fit state for train-window preprocessing. The newly defined
feature-block grammar is:

- `feature_block_set`
- `target_lag_block`
- `x_lag_feature_block`
- `factor_feature_block`
- `level_feature_block`
- `rotation_feature_block`
- `temporal_feature_block`
- `feature_block_combination`

These axes are the canonical Layer 2 language. Supported runtime slices now
read explicit blocks first for executor-family dispatch, fixed target lags,
fixed X lags, and static PCA factors. The coarse `feature_builder`,
`predictor_family`, `data_richness_mode`, and `factor_count` bridge remains
accepted as compatibility/provenance because those names define `Z`, not
estimator behavior. New manifests should read the preferred
`compatibility_source` key for that provenance; `source_bridge` remains as a
legacy manifest alias. Downstream importance artifacts should likewise prefer
`feature_runtime_builder` plus `legacy_feature_builder`; `feature_builder`
remains a compatibility alias for existing readers.

Evaluation decomposition uses `feature_representation` as the canonical
component name for these axes. The old component name `feature_builder` remains
accepted as a decomposition-plan alias.

## Layer 3: Forecast Generator

Owns all choices that generate forecasts:

- forecast generator family, currently exposed as compatibility axis `model_family`
- registered custom forecast generator/model
- baseline generator role assignment, currently exposed as compatibility axis
  `benchmark_family`
- direct vs iterated forecast generation
- forecast object, including mean/median/quantile
- training window, refit policy, min train size, training start rule
- model lag counts and horizon modelization
- validation split, hyperparameter search, tuning objective, budget
- estimator seed use, early stopping, and convergence handling

Canonical cleanup:

- `model_family` remains the public compatibility axis, but its canonical
  meaning is forecast generator family.
- `benchmark_family` remains the public compatibility axis, but its canonical
  meaning is baseline generator role assignment.
- benchmarks belong here because they produce forecasts; they are forecast
  generators assigned the benchmark/baseline role, not a separate model species.
- Layer 3 owns estimator seed use, early stopping, and convergence handling,
  but not all runtime discipline.

Runtime discipline is split by layer:

| Discipline | Owner |
|---|---|
| Experiment execution control: failure policy, compute mode, reproducibility mode, broad cache/checkpoint policy | Layer 0 |
| Data timing: vintage, release lag, contemporaneous information, and availability | Layer 1 |
| Estimator training behavior: validation split, tuning, early stopping, convergence, model-specific seed use | Layer 3 |

Layer 3 consumes `Z_train`/`Z_pred` from Layer 2 and fits/predicts with a model
or benchmark. Its canonical contract is:

```text
fit_predict(forecast_generator, Layer2Representation, training_spec) -> forecast_payload
```

`forecast_payload_v1` contains `y_pred`, `selected_lag`, `selected_bic`, and
optional `tuning_payload`. The runtime coerces legacy executor dictionaries into
this contract and records `forecast_payload_contract=forecast_payload_v1` in
forecast metadata when available.

Typed scalar payload families are also Layer 3 contracts. Direction, interval,
and density currently wrap supported scalar point generators and emit explicit
payload artifacts. Their detailed producer/consumer/status definitions live in
`layer_contract_ledger.md`.

Layer 3 may validate that a selected forecast generator can consume the Layer 2
handoff, but it must not decide how `Z` was built. The following remain Layer 2
facts even when old recipes pass them through training-shaped fields:

- `feature_builder`, `predictor_family`, `data_richness_mode`, and
  `factor_count`;
- `target_lag_block`, `x_lag_feature_block`, `factor_feature_block`,
  `level_feature_block`, `temporal_feature_block`, `rotation_feature_block`,
  and `feature_block_combination`;
- missing/outlier/scaling/selection/normalization choices that change model
  inputs;
- custom feature blocks and custom matrix preprocessors.

The main split is fixed target-lag features versus AR order selection.
`target_lag_block=fixed_target_lags` is Layer 2 feature construction. AR BIC
lag selection is Layer 3 estimator behavior.

During migration, legacy recipe paths may still place Layer 2 axes near
training settings. Runtime dispatch should prefer the Layer 2 block spec when
it is present, and keep old fields only as compatibility/provenance aliases.

## Layer 4: Evaluation

Owns scoring and reporting of forecasts:

- primary metric
- point, relative, density, direction, economic metric families
- aggregation over time, horizon, and target
- ranking and report style
- regime-specific evaluation subsets
- `oos_period`, because recession/expansion-only OOS selection is an evaluation
  subset filter over already-built forecast origins

Layer 4 should not fit models, transform predictors/target, or run statistical tests.
During migration, `data_task_spec.oos_period` remains a compatibility mirror,
but `evaluation_spec.oos_period` is the canonical runtime input.

## Layer 5: Artifacts

Owns what gets saved:

- export format
- saved object set
- provenance field depth
- artifact granularity

## Layer 6: Inference

Owns statistical inference over forecast errors:

- equal predictive ability tests
- nested model tests
- multiple model tests
- density/interval tests
- direction tests
- residual diagnostics
- dependence correction / HAC policy
- test scope

## Layer 7: Interpretation

Owns interpretation after forecasts and metrics exist:

- importance method
- native/model-agnostic importance
- SHAP
- partial dependence
- local surrogate
- grouped/temporal/stability importance
- importance output style

## Compatibility

Existing recipes may still place migrated axes at their old layer path. The
compiler accepts that during migration. New docs, examples, and generated recipes
should move toward canonical ownership.

For official dataset transformations, canonical Layer 1 axes are
`official_transform_policy` and `official_transform_scope`. The older Layer 2
t-code fields remain compatibility inputs for legacy recipes. New generated
recipes should use the Layer 1 axes; the compiler derives any runtime
`PreprocessContract` fallback fields from those Layer 1 choices, and execution
reads `data_task_spec` first. `data_task_spec["official_transform_source"]` and
runtime t-code reports record whether the choice came from canonical Layer 1
axes or from the legacy `PreprocessContract` bridge.
