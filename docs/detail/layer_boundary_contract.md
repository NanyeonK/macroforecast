# Layer Boundary Contract

Date: 2026-04-22

This document is the canonical boundary contract for the recipe layers after the
Layer 1/2 cleanup decision.

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

These axes are registry-only in the current runtime. Existing executable paths
still use the coarse `feature_builder`, `predictor_family`,
`data_richness_mode`, and `factor_count` bridge, but those bridge names are
Layer 2 concepts because they define `Z`, not estimator behavior.

## Layer 3: Forecast Generator

Owns all choices that generate forecasts:

- model family
- benchmark family
- direct vs iterated forecast generation
- forecast object, including mean/median/quantile
- training window, refit policy, min train size, training start rule
- model lag counts and horizon modelization
- validation split, hyperparameter search, tuning objective, budget
- model seed, early stopping, convergence, cache, checkpointing, execution backend

Benchmarks belong here because they produce forecasts.

Layer 3 consumes `Z_train`/`Z_pred` from Layer 2 and fits/predicts with a model
or benchmark. During migration, legacy recipe paths may still place
`feature_builder`, `predictor_family`, `data_richness_mode`, and `factor_count`
near training settings because old executors use those names for dispatch. Their
canonical ownership is Layer 2 because they define feature representation, not
estimator behavior.

## Layer 4: Evaluation

Owns scoring and reporting of forecasts:

- primary metric
- point, relative, density, direction, economic metric families
- aggregation over time, horizon, and target
- ranking and report style
- regime-specific evaluation subsets

Layer 4 should not fit models, transform X/y, or run statistical tests.

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
reads `data_task_spec` first.
