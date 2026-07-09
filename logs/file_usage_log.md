# File Usage Log

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
