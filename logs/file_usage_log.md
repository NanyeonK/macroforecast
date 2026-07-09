# File Usage Log

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
