# File Usage Log

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
