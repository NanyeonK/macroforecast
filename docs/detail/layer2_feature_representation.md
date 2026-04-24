# Layer 2 Feature Representation

Date: 2026-04-24

Layer 2 is the research preprocessing and feature-representation layer. Its job
is to turn the Layer 1 official data frame into the model input matrices used by
Layer 3.

See `layer2_revision_plan.md` for the step-by-step migration plan from the
current bridge to the explicit feature-block runtime.

## Contract

Layer 1 provides:

- `H`: the available level/raw-style panel after source loading, sample
  restriction, information-set rules, release-lag handling, variable-universe
  filtering, and any raw-source missing/outlier repair selected before official
  transforms;
- `X`: the official transformed predictor frame when dataset/FRED-MD/QD
  transformation codes are applied;
- `target`: the target series and horizon-aligned target construction inputs
  (papers may write this as `Y_t` or `y_{t+h}`);
- provenance for availability, official transforms, and raw-source repair.

Layer 2 must produce, for each training window and forecast origin:

- `Z_train`: the feature matrix fit by Layer 3;
- `Z_pred`: the one-origin or batch prediction feature matrix;
- `feature_names`: stable names for every column in `Z`;
- `feature_block_metadata`: which blocks generated each feature;
- fit state for any imputer, scaler, selector, factor extractor, rotation, or
  custom preprocessor that was fit on the training window.

This is also where generic `Z` unification belongs. Raw-panel, factor-panel,
and target-lag-only builders should all lower into one Layer 2 representation
payload before Layer 3 sees the data.

Any operation that estimates parameters from data must be recursive or
train-only for the relevant forecast origin. Full-sample transformations are
allowed only when a recipe is explicitly a fixed replication profile and the
provenance marks that choice.

## What Layer 2 Owns

Layer 2 owns four types of decisions.

| Group | Canonical questions | Existing bridge axes |
|---|---|---|
| Frame conditioning | How are post-official-frame missing values, outliers, scaling, filters, and target transforms handled? | `x_missing_policy`, `x_outlier_policy`, `scaling_policy`, `additional_preprocessing`, `target_transform`, `target_transformer` |
| Target representation | Which target scale or horizon target is handed to the forecast generator? | `horizon_target_construction`, `target_transform`, `target_normalization`, `target_transformer` |
| Feature-block construction | Which blocks are built from `H`, `X`, and target history before forecasting? | `feature_builder`, `x_lag_creation`, `dimensionality_reduction_policy`, `feature_selection_policy` |
| Block composition | Which blocks are included in `Z`, and how are they concatenated or substituted? | `predictor_family`, `data_richness_mode`, `feature_grouping` |
| Representation dimensions and leakage discipline | How many factors/lags/features are used, and where are transforms fit? | `factor_count`, `preprocess_fit_scope`, `separation_rule` |

Layer 2 does not own model family, benchmark family, direct/iterated forecast
generation, validation split, hyperparameter search, scoring metrics,
statistical tests, or feature-importance interpretation.

## Representation Handoff

The canonical Layer 2 output is not just a numeric matrix. It is one
representation handoff contract with:

- `Z_train`, `y_train`, `Z_pred`;
- stable `feature_names`;
- block provenance and block roles;
- train-window fit state;
- alignment and leakage metadata.

If runtime code still branches on legacy `feature_builder` families after this
handoff point, that is compatibility debt. The ownership is still Layer 2: the
builder path must be normalized before Layer 3 estimator code runs.

## Feature-Block Grammar

Runtime dispatch now prefers the explicit Layer 2 feature-block grammar for
the first raw-panel/autoregressive route decision. The old `feature_builder`
names remain accepted as compatibility aliases and as fallback source
provenance for old recipes.

| Axis | Values | Meaning |
|---|---|---|
| `feature_block_set` | `target_lags_only`, `transformed_x`, `transformed_x_lags`, `factor_blocks_only`, `factors_plus_target_lags`, `high_dimensional_x`, `selected_sparse_x`, `level_augmented_x`, `rotation_augmented_x`, `mixed_blocks`, `custom_blocks`, `legacy_feature_builder_bridge` | Top-level recipe for which blocks should form `Z`. `legacy_feature_builder_bridge` is retained only as a compatibility value for unknown old bridge recipes. |
| `target_lag_block` | `none`, `fixed_target_lags`, `ic_selected_target_lags`, `horizon_specific_target_lags`, `custom_target_lags` | Target-history features built from the target series. |
| `target_lag_selection` | `none`, `fixed`, `ic_select`, `cv_select`, `horizon_specific`, `custom` | Target-language replacement for public Layer 2 lag-selection provenance; legacy `y_lag_count` remains accepted for Layer 3/model-order compatibility. |
| `x_lag_feature_block` | `none`, `fixed_x_lags`, `variable_specific_x_lags`, `category_specific_x_lags`, `cv_selected_x_lags`, `custom_x_lags` | Lagged predictor features built from `X`. |
| `factor_feature_block` | `none`, `pca_static_factors`, `pca_factor_lags`, `supervised_factors`, `custom_factors` | Reduced-rank/factor features built from `X`. |
| `level_feature_block` | `none`, `target_level_addback`, `x_level_addback`, `selected_level_addbacks`, `level_growth_pairs` | Level or level-growth add-back features built from `H` and target history. |
| `rotation_feature_block` | `none`, `marx_rotation`, `maf_rotation`, `moving_average_rotation`, `custom_rotation` | Rotated features such as moving-average rotations of `X` or factors. |
| `temporal_feature_block` | `none`, `moving_average_features`, `rolling_moments`, `local_temporal_factors`, `volatility_features`, `custom_temporal_features` | Local time-series features built within each training window. |
| `feature_block_combination` | `replace_with_blocks`, `append_to_base_x`, `append_to_target_lags`, `concatenate_named_blocks`, `custom_combiner` | How selected blocks are assembled into `Z`. |

Operational support is currently narrow:

- `target_lag_block=none` and `fixed_target_lags` are operational. Fixed
  target-lag matrix composition reads `target_lag_block` directly; legacy
  `target_lag_selection`, `y_lag_count`, and `factor_ar_lags` remain accepted
  as fallback/provenance fields.
- `target_lag_selection=none` and `fixed` are operational Layer 2 names; IC,
  CV, horizon-specific, and custom lag selection remain registry-only.
- `x_lag_feature_block=none` and `fixed_x_lags` are operational. Fixed X-lag
  matrix composition reads `x_lag_feature_block` directly; legacy
  `x_lag_creation` remains accepted as fallback. Variable, category, CV, and
  custom X-lag blocks remain registry-only.
- Simultaneous fixed target-lag and raw-panel X-block composition is
  operational. In direct raw-panel rows, `target_lag_1` is the target observed
  at the forecast origin, `target_lag_2` is the previous target value, and so
  on. Fixed target lags can compose with fixed X lags and with
  `pca_static_factors`; in the factor case, PCA is fit on the X-side block and
  target lags are concatenated after the factor scores.
- `factor_feature_block=none`, `pca_static_factors`, `pca_factor_lags`,
  `supervised_factors`, and registered `custom_factors` are operational.
  `pca_static_factors` now drives matrix composition directly; `pca_factor_lags`
  appends lagged factor-score blocks; `supervised_factors` uses train-window PLS
  factors; registered custom factor blocks use `custom_feature_block_callable_v1`.
  Old
  `feature_builder=factor_pca` / `factors_plus_AR` and raw-panel
  `dimensionality_reduction_policy=pca` / `static_factor` remain accepted
  compatibility paths. Runtime writes a factor fit-state artifact containing
  stable factor/loadings provenance.
- `level_feature_block=none`, `target_level_addback`, `x_level_addback`,
  `selected_level_addbacks`, and `level_growth_pairs` are operational for
  raw-panel feature runtimes.
  `target_level_addback` appends the observable target level at the feature row
  date (`target_t`) and at the prediction origin (`target_origin`).
  `x_level_addback` appends raw-level `H` values preserved after Layer 1 raw
  missing/outlier handling and before official transforms/T-codes, using
  `{predictor}_level` public names. `selected_level_addbacks` applies the same
  source/alignment rule only to `leaf_config.selected_level_addback_columns`.
  `level_growth_pairs` records existing transformed predictor columns paired
  with selected raw-level counterparts from `leaf_config.level_growth_pair_columns`.
- `temporal_feature_block=none`, `moving_average_features`,
  `rolling_moments`, `local_temporal_factors`, and `volatility_features` are
  operational for raw-panel feature runtimes. The current lowered slices append
  trailing 3-period moving-average `{predictor}_ma3`, moment
  `{predictor}_mean3` / `{predictor}_var3`, local temporal factor
  `local_temporal_factor_mean3` / `local_temporal_factor_dispersion3`, or
  volatility `{predictor}_vol3` features. These deterministic append blocks can
  compose with fixed predictor lags and `moving_average_rotation` in raw-panel
  runtimes; supported static PCA factor composition is open through explicit
  composers, while non-PCA or factor-of-augmented-panel semantics remain
  gated. Local temporal
  factors are deterministic row-wise cross-sectional summaries of the active
  predictor panel, smoothed over the trailing 3 feature rows; they are not
  learned PCA/static factors.
- `temporal_feature_block=custom_temporal_features` is executable when the
  recipe names a registered callable through `custom_feature_blocks` or
  `custom_temporal_feature_block`. It is not the same contract as
  `custom_preprocessor`: the existing custom preprocessor is a broad matrix hook
  and does not guarantee block-local feature names, fit-state provenance, or
  leakage metadata.
- `rotation_feature_block=none`, `moving_average_rotation`, and `marx_rotation`
  are operational for raw-panel feature runtimes. `none` records explicit
  no-rotation provenance. `moving_average_rotation` appends deterministic
  trailing 3- and 6-period moving-average rotations of each active predictor as
  `{predictor}_rotma3` and `{predictor}_rotma6`, using only row-date /
  prediction-origin history, and can compose with fixed predictor lags and
  deterministic temporal append blocks. `marx_rotation` requires
  `leaf_config.marx_max_lag`, builds the cumulative lower-triangular
  lag-polynomial rotation, and replaces the source lag-polynomial basis in
  final `Z` by default. It now also supports `marx_then_factor`, where static
  PCA factors are fit on the MARX basis before any target-lag append, plus
  append composition with fixed X lags or deterministic temporal blocks via
  `feature_block_combination=append_to_base_x`.
- Advanced rotation values are explicit boundaries, not aliases for the generic
  primitive. `maf_rotation` remains registry-only until factor-score fit/apply
  state can compose with rotation blocks. `custom_rotation` remains registry-only
  under the `custom_feature_block_callable_v1` contract.
- The MARX composer is defined by `lag_polynomial_rotation_contract_v1`. Its
  naming contract is
  `{predictor}_marx_ma_lag1_to_lag{p}` for public feature names and
  `{predictor}__marx_ma_lag1_to_lag{p}` for runtime names, ordered by predictor
  and then rotation order. Its alignment contract is
  `Z_{i,p,t} = p^{-1} * sum_{j=1}^{p} X_{i,t-j}` for training rows and
  `X_{origin-1}, ..., X_{origin-p}` at prediction origin. Initial unavailable
  lags follow the package lag convention and are zero-filled before the sample
  start. Source lag columns must not be appended a second time when the MARX
  basis-replacement mode is active. The current composer supports raw-panel MARX
  basis replacement, `marx_then_factor` with static PCA factors, and named-block
  append to base `X` for fixed X-lag or deterministic temporal compositions.
  `factor_then_marx`, MAF rotation, unregistered custom rotation, and custom
  combiners remain explicitly gated.
- Feature selection is now operational for two explicit static-factor composer
  semantics when `factor_feature_block=pca_static_factors` (or the equivalent
  `dimensionality_reduction_policy` bridge) is active:
  `select_before_factor` first selects raw predictor columns within each train
  window and then estimates the static factor block on the selected panel;
  `select_after_factor` first estimates the static factor block, optionally
  appends target lags in the final `Z`, and then selects among the composed
  final columns. Broader factor/selection composition beyond static PCA still
  needs an explicit composer contract.

## Target Scale Contract

Layer 2 records `target_scale_contract_v1` for every recipe. The contract
separates:

- model target scale;
- forecast output scale;
- evaluation scale;
- inverse-transform policy;
- target-normalization fit scope.

The executable path now includes per-window target normalization
(`zscore_train_only`, `robust_zscore`, `minmax`, `unit_variance`), built-in
target-only inverse policies, `transformed_scale`, and `both` evaluation.
Prediction artifacts carry model, transformed, and original target-scale
columns, and metrics include scale-specific summaries. Custom inverse policies
remain gated.

## Target Representation Grammar

Coulombe et al. (2021) explicitly compare target construction choices in
addition to predictor transformations. In their notation, the direct approach
fits the average growth or difference target over steps 1 through `h`, while the
path-average approach fits each stepwise target separately and averages the
forecasts.

Layer 2 therefore owns the target representation choice:

| Axis | Values | Runtime status |
|---|---|---|
| `horizon_target_construction` | `future_target_level_t_plus_h`, `future_diff`, `future_logdiff`, `average_growth_1_to_h`, `average_difference_1_to_h`, `average_log_growth_1_to_h` | operational |
| `horizon_target_construction` | `path_average_growth_1_to_h`, `path_average_difference_1_to_h`, `path_average_log_growth_1_to_h` | registry-only |

Path-average target construction also requires Layer 3 support because the
forecast generator must fit multiple stepwise models and aggregate their
predictions. The target formula remains Layer 2; the multi-model forecast
execution protocol remains Layer 3. The package therefore records a
protocol-only `path_average_target_protocol_v1` payload for these choices:
Layer 2 lists the stepwise target formulas and equal-weight aggregation rule,
while the compiler keeps execution gated until Layer 3 can write per-step and
aggregate forecast artifacts.

## Mapping From Existing Bridge Names

The current coarse names map to the new language as follows:

| Current bridge | Feature-block interpretation |
|---|---|
| `feature_builder=autoreg_lagged_target` | target-lag block only; fixed target-lag construction now drives the supervised target-lag matrix path, with legacy fields retained as fallback/provenance. |
| `feature_builder=raw_feature_panel` | transformed or raw predictor panel block, chosen after Layer 1 official-frame policy and `predictor_family`. |
| `feature_builder=raw_X_only` | predictor panel block without target-lag features. |
| `feature_builder=factor_pca` | `feature_block_set=factor_blocks_only` plus a static factor feature block from the predictor panel. |
| `feature_builder=factors_plus_AR` | factor feature block plus target-lag block. |
| `data_richness_mode=target_lags_only` | `feature_block_set=target_lags_only`. |
| `data_richness_mode=factor_plus_lags` | `feature_block_set=factors_plus_target_lags`. |
| `data_richness_mode=full_high_dimensional_X` | `feature_block_set=high_dimensional_x`. |
| `data_richness_mode=selected_sparse_X` | `feature_block_set=selected_sparse_x`. |

This mapping is partly executable. Fixed target-lag, fixed X-lag, and static
PCA factor blocks can be selected directly in their supported runtime slices.
Joint composition that is not already represented by an executable runtime path
remains gated until the block composer has train-window fit/apply tests and
provenance.

Compiled manifests write `compatibility_source` as the preferred provenance key
for old bridge values that were accepted as input. The older `source_bridge`
manifest key remains as a compatibility alias for existing downstream readers.

## Boundary Cases

`y_lag_count` is legacy compatibility language and is split by meaning:

- if it selects AR/model order inside an estimator, it remains Layer 3;
- if it creates lagged target columns in `Z`, the Layer 2 provenance name is
  `target_lag_selection` plus `target_lag_block`.

`factor_ar_lags` is also split by meaning:

- target-lag feature construction next to factor blocks is recorded as
  `target_lag_count` in Layer 2 provenance;
- model-specific lag-order selection belongs to Layer 3.

`dimensionality_reduction_policy` and `factor_feature_block` are related but
not identical. The explicit `factor_feature_block` now owns static PCA matrix
composition in the supported slice; `dimensionality_reduction_policy` remains a
compatibility fallback. Future composer work will define factor blocks combined
with target lags, level add-backs, rotations, and custom blocks.

## Implementation Order

The Layer 2 cleanup migration is complete for supported fixed full/runtime
slices:

1. The legacy `feature_builder` bridge remains operational as input
   compatibility and provenance.
2. Compiled specs record feature-block provenance without changing unsupported
   runtime matrices.
3. Fixed `target_lag_block` and fixed `x_lag_feature_block` execute through
   explicit block paths with train-window alignment tests.
4. `factor_feature_block=pca_static_factors`, `pca_factor_lags`,
   `supervised_factors`, and registered `custom_factors` execute through
   train-window fit/apply paths with provenance.
5. Built-in level add-backs, deterministic temporal append blocks, registered
   custom blocks, `moving_average_rotation`, and MARX lag-polynomial rotation
   execute for raw-panel feature runtimes where their composition contracts are
   supported.
6. Built-in target normalization/inverse/evaluation-scale artifacts execute
   inside the rolling/expanding forecast loop.
7. Runtime dispatch, target-transformer gates, importance/custom hook
   contexts, and decomposition metadata now use block-derived feature runtime
   provenance for supported slices.

Remaining work is semantic feature-composer work, not bridge cleanup:
`factor_then_marx`, MAF rotation, custom combiners, feature selection over
non-PCA/custom/deterministic append blocks, and custom target inverse policies.

The detailed target contract for freely sweeping Layer 2 representations with
Layer 3 forecast generators is documented in
`layer2_layer3_sweep_contract.md`.
The current closed/open status is tracked in `layer2_closure_ledger.md`.
