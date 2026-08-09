# File Usage Log

## 2026-07-11 - ADD6 supervised PCA preselection stage

- `macroforecast/models/linear.py`: added strict
  `preselect_stage={"after_standardize", "raw_before_standardize"}` handling to
  `SupervisedPCARegressor`, `supervised_pca`, and `supervised_scaled_pca`;
  default `after_standardize` keeps the existing full-standardize-then-screen
  path, while raw-before mode screens raw factors and scales only selected
  factors plus required controls.
- `macroforecast/models/specs.py`: registered the non-tunable
  `preselect_stage` parameter on both supervised PCA model specs without adding
  it to search spaces.
- `tests/models/test_models.py`: added scale-sensitive raw-before screening,
  default-vs-explicit-after unchanged, validation, registry, and discarded
  predictor prediction coverage for both supervised PCA wrappers.
- `docs/reference/models.md`: regenerated with system `python3 -m tools.docgen`
  so the public signatures and parameter tables include `preselect_stage`.
- `CHANGELOG.md` and `logs/file_usage_log.md`: recorded this additive option
  and file-touch summary.

## 2026-07-10 - ADD4 leak-free predictor standardization scope

- `macroforecast/preprocessing/specs.py`: added canonical
  `standardize_scope="origin_available_predictors"` handling plus
  `include_current_predictor_rows=True` normalization, deferred fit-time
  scaling for the new scope, and per-origin predictor-only state fitting from
  available rows.
- `macroforecast/preprocessing/preprocess.py`: extended `standardize_panel(...)`
  with an opt-in origin-available predictor scope while preserving the old
  full-panel call path by default.
- `macroforecast/preprocessing/cache.py` and
  `macroforecast/forecasting/preprocessing_stage.py`: recorded and keyed
  configured standardization scope so fitted/prepared preprocessing caches do
  not cross scopes.
- `tests/preprocessing/test_preprocess.py` and
  `tests/preprocessing/test_fitted_cache.py`: added leak-proof sentinel,
  off-path unchanged, direct helper, alias, and cache-scope coverage.
- `docs/reference/preprocessing.md` and
  `docs/guide/concepts/preprocessing.md`: regenerated/updated preprocessing
  docs, including the target-standardization leak divergence.
- `CHANGELOG.md` and `logs/file_usage_log.md`: recorded this builder
  file-touch summary.

## 2026-07-10 - ADD3 pls score projection

- `macroforecast/models/linear.py`: added `score_projection` to `pls`, keeping
  the default sklearn-transform path and adding opt-in raw x-weight score
  projection with the existing external forecast head.
- `macroforecast/models/specs.py`: registered the non-tunable
  `score_projection` parameter with default `"transform"` and preserved existing
  PLS search spaces.
- `tests/models/test_models.py`: added PLS default-anchor, raw-score oracle,
  quadratic raw-head, validation, and registry coverage.
- `docs/reference/models.md` and `docs/guide/models/composite.md`: regenerated
  generated model docs for the new PLS option and generalized PLS wording.
- `CHANGELOG.md` and `logs/file_usage_log.md`: recorded this additive option
  and file-touch summary.

## 2026-07-10 - ADD2 ar_bic doc-drift regen

- `docs/reference/pipeline.md`: regenerated with `python3 -m tools.docgen`
  so the `DIRECT_POLICY_GUARD_MODELS` reference listing includes `ar_bic`.
- `docs/guide/model_policy_matrix.md`: regenerated with
  `python3 tools/gen_policy_matrix.py --out docs/guide` so `ar_bic` is marked
  guarded-unsupported for direct policies.
- `logs/file_usage_log.md`: recorded this builder doc-regeneration summary.

## 2026-07-10 - ADD2 ar_bic tester-block fixes

- `tests/models/test_ar_bic.py`: pinned the independently derived
  lag_square-BIC controlled-series expectation to selected lag `4` and
  coefficient-power forecast `2.6204213101651392`.
- `macroforecast/pipeline/spec.py`: added `ar_bic` to
  `DIRECT_POLICY_GUARD_MODELS` because it is target-only and lacks direct-policy
  support.
- `logs/file_usage_log.md`: recorded this builder fix summary.

## 2026-07-10 - ADD2 ar_bic

- `macroforecast/models/timeseries.py`: added the target-only `ar_bic`
  estimator with internal residual-variance AIC/AICc/BIC lag selection,
  selectable IC parameter counts, OLS/Yule-Walker/Burg/internal
  MATLAB-compatible backends, and iterated, direct-lag-projection, and
  coefficient-power forecast modes. Existing `ar` code paths were left intact.
- `macroforecast/models/specs.py`, `macroforecast/models/__init__.py`, and
  `macroforecast/__init__.py`: registered/exported `ar_bic` with conservative
  model-owned search spaces and top-level lazy access.
- `tests/models/test_ar_bic.py`: added builder coverage for reference
  BIC-lag/coefficient-power behavior, registry target-only routing, backend
  smoke paths, validation errors, and default-`ar` unchanged regression.
- `docs/reference/models.md`, `docs/reference/public_api.md`,
  `docs/reference/reference_verification.md`, `docs/guide/model_overview.md`,
  `docs/guide/model_policy_matrix.md`, and `docs/guide/models/timeseries.md`:
  regenerated generated model documentation after registering `ar_bic`.
- `CHANGELOG.md`: documented the additive `ar_bic` model and unchanged default
  `ar` behavior.
- `logs/file_usage_log.md`: recorded this builder file-touch summary.

## 2026-07-10 - ADD1 pcr

- `macroforecast/models/linear.py`: added `PCRRegressor` and the public `pcr`
  callable with predictor-only standardization, optional control
  residualization, and separate/joint squared-score heads.
- `macroforecast/models/__init__.py`, `macroforecast/__init__.py`, and
  `macroforecast/models/specs.py`: exported and registered `pcr` with
  scaled-PCA-style `n_components` search spaces.
- `tests/models/test_models.py`: added residualized-PCR reference tests,
  quadratic-head coverage, validation/nan-policy coverage, and public
  registry/export checks.
- `tests/models/test_ar_far_direct_projection.py`: added a deterministic FAR
  prediction regression anchor.
- `docs/reference/models.md`, `docs/reference/public_api.md`,
  `docs/reference/reference_verification.md`, `docs/guide/model_overview.md`,
  `docs/guide/model_policy_matrix.md`, and `docs/guide/models/composite.md`:
  regenerated generated model documentation after registering `pcr`.
- `CHANGELOG.md`: documented the additive `pcr` model and FAR unchanged anchor.
- `logs/file_usage_log.md`: recorded this builder file-touch summary.

## 2026-07-09 - FIX5 random_forest defaults

- `macroforecast/models/tree.py`: changed `random_forest` defaults to
  `n_estimators=500` and `max_features=1.0 / 3.0`, recording and passing
  `max_features` through to `RandomForestRegressor`.
- `macroforecast/models/specs.py`: updated registered `random_forest`
  defaults/parameter metadata and added `_RANDOM_FOREST_SPACES` so
  `extra_trees` and `quantile_regression_forest` keep the shared forest spaces.
- `tests/models/test_models.py`: added new-default metadata/estimator tests and
  explicit old-param compatibility coverage against sklearn all-feature splits.
- `tests/model_selection/test_search_specs.py`: pinned RF-specific search-space
  keys and confirmed `extra_trees`/QRF spaces remain unchanged.
- `tests/forecasting/test_forecasting.py`: added default-RF forecasting smoke
  and explicit `max_features` params preservation coverage.
- `docs/reference/models.md` and `docs/guide/models/tree.md`: regenerated after
  the RF signature/search-space metadata change.
- `CHANGELOG.md`: documented the DEFAULT CHANGED migration note.
- `logs/file_usage_log.md`: recorded this builder file-touch summary.

## 2026-07-09 - FIX2 parallel executor reliability

- `macroforecast/pipeline/run.py`: replaced parallel cell `executor.map` with
  submitted futures collected through a heartbeat timeout, records timeout or
  broken-pool failures via failed cells, preserves completed result-store writes,
  and shuts down failed pools without unbounded waits.
- `macroforecast/pipeline/spec.py`: added validated
  `parallel_cell_timeout` execution control with default `3600.0` seconds and
  explicit `None` opt-out for timeout detection.
- `tests/pipeline/test_parallel_executor_reliability.py`: added stress coverage
  for slow cells, killed workers, result-store resume after partial timeout,
  serial-vs-parallel numerical equality, and timeout validation.
- `docs/reference/pipeline.md`: regenerated public API reference for the new
  `PipelineSpec` field and `pipeline_spec(...)` argument.
- `CHANGELOG.md`: documented the reliability-only executor hardening and
  unchanged-number/result-store semantics.
- `logs/file_usage_log.md`: recorded this builder file-touch summary.

## 2026-07-09 - FIX4 horizon-dependent rolling window

- `macroforecast/window/core.py`: added optional rolling `size_rule` and
  `size_by_horizon` support, resolved against the injected test horizon during
  origin planning while preserving fixed rolling-window behavior.
- `tests/window/test_horizon_dependent_rolling.py`: added coverage for
  horizon-rule sizing, explicit per-horizon sizing, map-only default origins,
  fixed rolling origin/metadata regression, same-R deterministic OLS forecast
  equality, fixed forecast golden predictions, and validation errors.
- `docs/reference/window.md`: regenerated public API reference after the
  `EstimationWindow`, `estimation_rolling`, and `from_cutoffs` signatures
  changed.
- `CHANGELOG.md`: documented the additive horizon-dependent rolling-size API and
  fixed-window parity guarantee.
- `logs/file_usage_log.md`: recorded this builder file-touch summary.

## 2026-07-09 - FIX3 UCSV docs regeneration

- `docs/reference/models.md`: regenerated reference docs so the UCSV signature
  and parameter table include `initial_obs_log_vol_variance` and
  `initial_level_log_vol_variance`.
- `docs/reference/feature_engineering.md`: regenerated incidentally from the
  doc generator; this captures pre-existing pandas-repr drift
  (`pandas.DataFrame` -> `pandas.core.frame.DataFrame`).
- `docs/reference/preprocessing.md`: regenerated incidentally from the doc
  generator; this captures pre-existing pandas-repr drift
  (`pandas.DataFrame` -> `pandas.core.frame.DataFrame`).
- `logs/file_usage_log.md`: recorded this builder doc-regeneration summary.

## 2026-07-09 - FIX3 UCSV knobs

- `macroforecast/models/bayesian.py`: added UCSV initial-prior variance
  parameters for the observation and trend-innovation log-volatility states,
  threaded into the Gibbs sampler's log-volatility state initialization while
  preserving the current implicit `10.0` defaults and existing `random_state`.
- `macroforecast/models/specs.py`: exposed the new UCSV parameters in the model
  registry so model params and `Arm.params` can pass them through.
- `tests/models/test_standard_estimators.py`: added UCSV default-forecast anchor
  coverage, deterministic custom-prior coverage, registry exposure checks, and
  validation checks for positive prior variances.
- `CHANGELOG.md`: documented the additive UCSV knobs and defaults-unchanged
  behavior.
- `logs/file_usage_log.md`: recorded this builder file-touch summary.

## 2026-07-09 - FIX1 params pin

- `macroforecast/forecasting/policies/base.py`: implemented explicit-param
  pinning for model-owned default search and explicit `SearchSpec` selection in
  the shared forecast policy skeleton.
- `tests/forecasting/test_forecasting.py`: added acceptance coverage for pinned
  params, no-params default-search regression, and explicit
  `model_selection={name: None}` disablement.
- `tests/forecasting/_golden/runner_snapshot.parquet`: regenerated the runner
  golden fixture after verifying only explicit-params FAR arms changed.
- `tests/model_ensemble/test_model_ensemble.py`: updated the model-ensemble
  alias combination test to expect all-pinned selection to skip metadata and
  preserve explicit `params`.
- `CHANGELOG.md`: documented the bug fix and intended forecast changes for arms
  whose explicit params were previously overridden.
- `logs/file_usage_log.md`: recorded this builder file-touch summary.

## 2026-07-10 - MYPY housekeeping

- `macroforecast/data_analysis/summary.py`: added an explicit `np.ndarray`
  annotation to the Engle-Granger trend time vector so mypy can type-check the
  local variable; no behavior, number, or API change.
- `logs/file_usage_log.md`: recorded this builder typing-housekeeping summary.

## 2026-07-10 - ADD5 nan policy and score aggregation

- `macroforecast/preprocessing/preprocess.py`: added opt-in
  `nan_policy="zero_after_standardize"` and `standardize_nan_fill=0.0` alias for
  post-standardization non-finite fill in selected standardized columns while
  preserving default propagation behavior.
- `macroforecast/model_selection/{types.py,builders.py,splitters.py,runner.py,optimizers.py,search.py}`:
  added `score_aggregation` plumbing with default `mean_split` and opt-in
  `mean_fold` logical-fold pooling for validation scoring.
- `tests/preprocessing/test_preprocess.py` and
  `tests/model_selection/test_select_params.py`: added option behavior,
  validation, and defaults-unchanged regression coverage.
- `docs/reference/{preprocessing.md,model_selection.md,custom/custom_window_selection_forecasting.md}`:
  regenerated reference docs with system `python3`.
- `CHANGELOG.md`: documented the additive options and unchanged-default
  guarantees.
- `logs/file_usage_log.md`: recorded this builder file-touch summary.

## Clark-West for forecast combinations (fix/cw-for-combinations)

- `macroforecast/pipeline/spec.py`: added `CombinationContender.nested_in_benchmark`
  and documented when it is licensed.
- `macroforecast/pipeline/evaluate.py`: declared combinations join the CW-eligible
  set; the previously silent NaN case warns.
- `tests/pipeline/test_cw_for_combinations.py`: new -- field/default, unchanged
  default behavior, CW emitted when declared, statistic matched against a
  hand-computed `clark_west_test`, and the warning.
- `docs/guide/concepts/evaluation.md`: stated the arm/combination CW rule.
- `docs/reference/{pipeline.md,models.md}`: regenerated with system `python3`
  (`models.md` was already stale from the merged `gtvp`/`variable_importance`
  and `nn` early-stopping changes).
- `CHANGELOG.md`, `logs/file_usage_log.md`: recorded.

## Restore a green CI (fix/star-attr-annotations)

- `macroforecast/models/_mrf_reference.py`: matplotlib import moved into the two
  plotting methods that use it.
- `macroforecast/models/tree.py`: dropped the matplotlib `optional_import` gate.
- `macroforecast/models/timeseries.py`: annotated `_STAR`'s seven attributes;
  widened the `predict` guard.
- `tools/docgen/renderer.py`: canonicalize public pandas class paths.
- `docs/reference/models.md`: regenerated (stale since the gtvp/nn merges).
- `tests/models/test_models.py`: lazy-failure case matplotlib -> joblib; removed
  three matplotlib `importorskip` gates; relaxed the PLS literal pin.
- `tests/models/test_ar_bic.py`: relaxed the forecast pin to rtol=1e-9.
- `tests/mc/test_dm_size.py`: marked the n50-h1-none case as a known
  distortion (strict xfail), matching the file's existing h=4 treatment.
- `CHANGELOG.md`, `logs/file_usage_log.md`: recorded.

## Target-only fit samples (fix/hist-mean-target-scope)

- `macroforecast/forecasting/selection_stage.py`: added `_is_target_only`;
  `_align_feature_xy`/`_filter_xy_to_target_availability` take `target_only`.
- `macroforecast/forecasting/runner.py`: `_slice_feature_set` takes `target_only`;
  both fit-sample call sites pass it.
- `macroforecast/forecasting/policies/direct.py`: threads the flag.
- `tests/forecasting/test_target_only_fit_sample.py`: new — benchmark invariance
  to a predictor gap, the prevailing-mean value, the supervised complement, and
  NaN targets still dropped.
- `CHANGELOG.md`, `logs/file_usage_log.md`: recorded.

## ols rank-deficiency warning (fix/ols-collinear-warn)

- `macroforecast/models/linear.py`: added `_warn_if_rank_deficient`; `ols` calls it
  after fitting.
- `tests/models/test_ols_rank_deficiency.py`: new -- duplicated column, dummies
  summing to the intercept, no false positive on a healthy or merely
  ill-conditioned design, predictions untouched, message content, and the
  `positive=True` path that has no singular values.
- `CHANGELOG.md`, `logs/file_usage_log.md`: recorded.

## mypy follow_imports (fix/mypy-follow-imports)

- `pyproject.toml`: `follow_imports` skip -> normal; enabled
  `no_implicit_optional`, `strict_equality`, `warn_unused_ignores`; recorded the
  measured cost of the two flags left off.
- `macroforecast/models/tvp.py`: `vol="GARCH"`; dropped a stale `type: ignore`.
- `macroforecast/feature_engineering/specs.py`: `order` -> `sort_order` for the
  argsort index.
- `macroforecast/data_analysis/summary.py`: three `cast` calls after existing
  runtime validation; dropped a stale `type: ignore`.
- `macroforecast/data/panel.py`: removed an always-true guard.
- `macroforecast/feature_engineering/feature_selection.py`,
  `macroforecast/interpretation/core.py`: dropped stale `type: ignore` comments.
- `CHANGELOG.md`, `logs/file_usage_log.md`: recorded.

## Feature-fit store tier (perf/feature-store-tier)

- `macroforecast/forecasting/feature_stage.py`: added `_fit_panel_fingerprint`
  and `_feature_store_key`; `_fitted_feature_builder_for_origin` takes
  `feature_store` and consults/populates it on an in-memory miss.
- `macroforecast/forecasting/runner.py`: both call sites forward
  `preprocessing_store`.
- `tests/forecasting/test_feature_store_tier.py`: new -- forecasts unchanged
  with the store, a warm store serves every fit, a same-shaped different dataset
  is not served a stored fit, fingerprint separates content at equal shape, key
  depends on both parts.
- `CHANGELOG.md`, `logs/file_usage_log.md`: recorded.

## far PCA convention (fix/far-pca-scale)

- `macroforecast/models/timeseries.py`: `_FAR` takes `scale`; added
  `_standardizer` / `_prepare_pca_input` and routed all three PCA sites (direct
  fit, iterated fit, transform) through them; `far()` exposes and documents it.
- `macroforecast/models/specs.py`: `far` declares `scale` in `default_params`
  and `parameters`.
- `tests/models/test_far_pca_scale.py`: new -- default reproduces covariance
  PCA, `scale=True` reproduces correlation PCA, the two disagree, covariance PC1
  is dominated by the large series, zero-variance column is safe, both fit paths
  honour the flag.
- `docs/reference/models.md`, `docs/guide/models/factor.md`: regenerated.
## Multiple-testing adjustment (feat/multiple-testing)

- `macroforecast/tests.py`: added `MULTIPLE_TESTING_METHODS`, `adjust_pvalues`
  (bonferroni/holm/bh) and `romano_wolf_pvalues` (block-bootstrap step-down);
  exported all three.
- `macroforecast/pipeline/evaluate.py`: `significance_table` collects the
  per-contender loss differentials and calls `_apply_multiple_testing`, which
  appends `<test>_p_adj` per `(target, horizon)` family.
- `macroforecast/pipeline/spec.py`: `multiple_testing` is validated against
  `MULTIPLE_TESTING_METHODS` instead of raising for every value.
- `tests/pipeline/test_multiple_testing.py`: new -- closed-form adjustments
  against their definitions, NaN excluded from the family, conservatism
  ordering, unknown method refused, Romano-Wolf keeps the genuine winner /
  is no stricter than Holm / is deterministic, the pipeline emits the column,
  no method leaves the report untouched, and all four adjust the same family.
- `CHANGELOG.md`, `logs/file_usage_log.md`: recorded.

## Security and hygiene (chore/security-and-hygiene)

- `macroforecast/models/persistence.py`: `load_fit` docstring warns that
  unpickling executes code.
- `macroforecast/preprocessing/cache.py`: the S301 suppression states its trust
  assumption.
- `SECURITY.md`: new — reporting route, trust assumptions, scope.
- `macroforecast/pipeline/spec.py`: `by` / `primary_axis` documented as reserved.
- `.gitignore`: removed `macrocast` rename residue.
- `docs/reference/models.md`: regenerated for the new docstring.
## Architecture guard (chore/architecture-guard)

- `macroforecast/window/policy.py`: `apply_to` refuses any non-default value;
  the field docstring records that nothing reads it.
- `tests/architecture/test_import_boundaries.py`: new -- layer map derived from
  actual imports, ratchet against new upward imports, known-exception list that
  fails if it goes stale, and a check that no package is unclassified.
- `docs/architecture.md`: new -- layers, why they matter, known exceptions, and
  two recorded [GAP]s (task resolution spread across four files; evaluation
  loading data).
- `docs/index.md`: architecture added to the toctree.
## Stateful custom preprocessing (fix/449-stateful-custom-preprocess)

- `macroforecast/preprocessing/specs.py`: `custom_preprocess_step` takes
  `fit_func`/`transform_func`/`row_local`; `_fit_custom_preprocess_states` runs
  fit callables on the estimation window; `FittedPreprocessor.custom_step_states`
  carries the result to transform; `_reject_unrestricted_fit_window_steps`
  replaces the old warning.
- `tests/preprocessing/test_custom_step_contract.py`: new -- refusal message
  names every way out, stateful step is future-proof, row-local still works,
  `origin_available` unchanged, state fitted exactly once, builder refuses
  incoherent combinations.
- `docs/guide/concepts/preprocessing.md`: the "keep them row-local" warning
  replaced with the contract and worked examples.
- `CHANGELOG.md`, `logs/file_usage_log.md`: recorded.

## Interpretation audit (audit/446-interpretation)

- `macroforecast/interpretation/core.py`: `_coerce_custom_table` distinguishes a
  columnar mapping from a scalar one; `custom_interpretation` docstring records
  the full contract and its non-guarantees.
- `tests/interpretation/test_attribution_oracles.py`: new -- native attribution
  oracles and the custom-model dispatch check.
- `tests/interpretation/test_shapley_oracles.py`: new -- efficiency, dummy,
  linearity, permutation stability, aggregated-vs-per-row agreement, constant
  model.
- `tests/interpretation/test_custom_interpretation_contract.py`: new -- call
  signature, accepted return shapes (including the fixed columnar mapping),
  attached schema, and the explicit non-guarantee.
- `docs/reference/interpretation.md`: regenerated for the new docstring.
- `CHANGELOG.md`, `logs/file_usage_log.md`: recorded.

