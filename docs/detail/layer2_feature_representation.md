# Layer 2 Feature Representation

Date: 2026-04-22

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

## Feature-Block Grammar

Runtime dispatch now prefers the explicit Layer 2 feature-block grammar for
the first raw-panel/autoregressive route decision. The old `feature_builder`
names remain accepted as compatibility aliases and as fallback source
provenance for old recipes.

| Axis | Values | Meaning |
|---|---|---|
| `feature_block_set` | `legacy_feature_builder_bridge`, `target_lags_only`, `transformed_x`, `transformed_x_lags`, `factors_plus_target_lags`, `high_dimensional_x`, `selected_sparse_x`, `level_augmented_x`, `rotation_augmented_x`, `mixed_blocks`, `custom_blocks` | Top-level recipe for which blocks should form `Z`. |
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
- Simultaneous target-lag and X-lag block composition is not executable yet:
  fixed target lags are supported by the standalone target-lag runtime, while
  fixed X lags are supported by macro-X panel runtimes. A composer that joins
  both blocks into one `Z` has not landed.
- `factor_feature_block=none` and `pca_static_factors` are operational.
  `pca_static_factors` now drives matrix composition directly; old
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
  runtimes; factor bridges remain gated until the explicit block composer
  defines append-to-factors vs factor-of-augmented-panel semantics. Local temporal
  factors are deterministic row-wise cross-sectional summaries of the active
  predictor panel, smoothed over the trailing 3 feature rows; they are not
  learned PCA/static factors.
- `temporal_feature_block=custom_temporal_features` remains registry-only. It
  is not the same contract as `custom_preprocessor`: the existing custom
  preprocessor is a broad matrix hook and does not guarantee block-local
  feature names, fit-state provenance, or leakage metadata. Operational custom
  temporal blocks need a callable contract that returns train/pred temporal
  feature frames plus names and provenance before they can enter `Z`.
- `rotation_feature_block=none`, `moving_average_rotation`, and `marx_rotation`
  are operational for raw-panel feature runtimes. `none` records explicit
  no-rotation provenance. `moving_average_rotation` appends deterministic
  trailing 3- and 6-period moving-average rotations of each active predictor as
  `{predictor}_rotma3` and `{predictor}_rotma6`, using only row-date /
  prediction-origin history, and can compose with fixed predictor lags and
  deterministic temporal append blocks. `marx_rotation` requires
  `leaf_config.marx_max_lag`, builds the cumulative lower-triangular
  lag-polynomial rotation, and replaces the source lag-polynomial basis in
  final `Z`.
- Advanced rotation values are explicit boundaries, not aliases for the generic
  primitive. `maf_rotation` remains registry-only until factor-score fit/apply
  state can compose with rotation blocks. `custom_rotation` remains registry-only
  until a block-local callable contract returns train/pred rotation feature
  frames, stable names, fit-state provenance, and leakage metadata.
- The MARX composer is defined by `lag_polynomial_rotation_contract_v1`. Its
  naming contract is
  `{predictor}_marx_ma_lag1_to_lag{p}` for public feature names and
  `{predictor}__marx_ma_lag1_to_lag{p}` for runtime names, ordered by predictor
  and then rotation order. Its alignment contract is
  `Z_{i,p,t} = p^{-1} * sum_{j=1}^{p} X_{i,t-j}` for training rows and
  `X_{origin-1}, ..., X_{origin-p}` at prediction origin. Initial unavailable
  lags follow the package lag convention and are zero-filled before the sample
  start. Source lag columns must not be appended a second time when the MARX
  basis is active. MARX currently cannot compose with external X-lag, temporal,
  or factor blocks until the explicit block composer exists.
- Feature selection currently applies only to raw predictor blocks. It cannot
  be combined with factor blocks or dimensionality reduction until the package
  defines selection-before-factor vs selection-after-factor semantics.

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
| `feature_builder=factor_pca` | factor feature block from the predictor panel. |
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

A safe implementation order is:

1. Keep current `feature_builder` bridge operational.
2. Add feature-block provenance to compiled specs without changing runtime
   matrices.
3. Implement fixed `target_lag_block` and fixed `x_lag_feature_block` with
   train-window alignment tests.
4. Implement `factor_feature_block=pca_static_factors` with recursive factor
   fit/apply tests and loadings provenance.
5. Extend `level_feature_block` beyond whole-panel add-backs, then implement
   deterministic non-none `rotation_feature_block` primitives such as
   `moving_average_rotation` before learned or custom rotations.
6. Retire runtime dispatch from coarse `feature_builder` names in slices:
   first route the raw-panel/autoregressive executor choice from explicit
   blocks, then move fixed X-lag matrix composition to `x_lag_feature_block`,
   then move fixed target-lag composition to `target_lag_block`, then move PCA
   static-factor matrix composition to `factor_feature_block`, then move
   importance/custom hook contexts to the block-derived runtime name. Remaining
   work is compiler gate wording, docs, and true joint-block composers.
