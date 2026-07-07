# A5 Screening + Selection Logging Progress

## Investigation
- Branch/worktree confirmed: `feat/screening-selector-logging` in `/home/nanyeon99/project/mf-a5-screening`.
- Read `.dev-notes/lane_ground_rules.md`; lane requires explicit staging, bounded pytest invocations, mypy, docs reference regeneration/check if public APIs change, and `CHANGELOG.md` under `[Unreleased]`.
- Existing supervised-PCA screening internals:
  - `macroforecast/models/linear.py:1723` calls `_preselect_columns(...)` before component extraction.
  - `macroforecast/models/linear.py:1990` implements `_preselect_columns`: `hard_tstat` keeps `abs(_marginal_t_stats) > t_threshold`; `elastic_net` keeps nonzero coefficients; empty selections fall back to all columns.
  - `macroforecast/models/linear.py:2011` implements marginal t-stats with a constant-only bivariate regression.
  - `macroforecast/models/linear.py:2027` implements iterative supervised-PCA component selection by absolute residual correlation, capped by `n_selected` and `min_abs_corr`.
- Feature-step carrier: `macroforecast/feature_engineering/specs.py` already has target-aware fit-window state classes for PLS/SIR/feature selection and records fitted step metadata on every origin. Decision: implement predictor screening as another target-aware fit-window feature step so leakage discipline matches existing feature-stage cadence.
- Checkpoint carrier: `macroforecast/forecasting/checkpoint.py` writes one scalar forecast parquet per origin under `<checkpoint>/<cell>/h<h>/origin_<pos>.parquet`. Decision: keep selection history out of the lean forecast parquet; write compact per-origin JSONL sidecars next to those parquet files.
- Rescore carrier: `macroforecast/pipeline/rescore.py` reconstructs reports from checkpoint dirs and records `provenance["rescored_from"]`. Decision: expose `selection_history(report_or_path)` that can read either live reports with `spec.checkpoint_dir`, rescored reports via `rescored_from`, or a checkpoint path.
- Target transforms: `macroforecast/feature_engineering/targets.py` delegates `average_*` to `_average_future_path`. Decision: add `log_average_value` as `log(mean(value[t+1..t+h]))`, with positivity gating before log.
- Preprocessing imputation: direct `reprocess` applies `impute` before `standardize`; fit-window `PreprocessSpec` applies fitted imputation before fitted standardization. Decision: add `impute="zero"` to both direct and fit-window state paths as missing-to-zero before standardization, consistent with existing stage order.

## Decisions
- Public screen step will be `predictor_screen(...)`, matching the workplan signature and returning a reusable feature-step mapping. It will support `method={"t_stat","delta_r2","lasso","elastic_net"}`, `threshold`, `top_k`, `min_k`, and `controls`.
- Existing supervised-PCA public behavior must remain byte-identical; regression pins will compare `supervised_pca` and `supervised_scaled_pca` outputs before/after extraction.
- Selection history is opt-in through pipeline/runner plumbing and requires a checkpoint directory; default runs write no sidecars.
- Selection sidecar format: `origin_<pos>_selection.jsonl` next to each completed `origin_<pos>.parquet`. The writer emits the sidecar before the forecast parquet; the loader ignores sidecars without a matching forecast parquet so interrupted origins do not surface as completed history.
- Public loader/API: `selection_history(report_or_path)` and `selection_frequency_table(...)`; reports use exact spec-described checkpoint cells when possible, while raw paths infer labels from `<target>__<arm>/h<h>/`.

## Gate Log
- Passed: syntax compile for changed runtime modules with `~/project/macroforecast/.venv/bin/python -m py_compile ...`.
- Passed: focused new tests for feature screening, target transform, zero imputation, SPCA helper identity, and selection history.
- Passed: affected pytest groups:
  - `tests/feature_engineering/test_features.py` (88 passed)
  - `tests/preprocessing/test_preprocess.py` (35 passed)
  - `tests/pipeline/test_rescore.py tests/pipeline/test_run_arms_stage1.py tests/pipeline/test_selection_history.py` (21 passed)
  - `tests/forecasting/test_checkpoint.py tests/forecasting/test_single_horizon_checkpoint_collision.py` (19 passed)
- Passed: `~/project/macroforecast/.venv/bin/python -m tools.docgen docs/reference` and `--check docs/reference`.
- Passed: `~/project/macroforecast/.venv/bin/python -m mypy macroforecast` (clean after changing the sidecar writer argument to `Sequence[Mapping[str, Any]]`; rerun clean after the final provenance lookup hardening).
- Passed final rerun: `tests/pipeline/test_selection_history.py` (5 passed) and `tests/forecasting/test_checkpoint.py tests/forecasting/test_single_horizon_checkpoint_collision.py` (19 passed).
- Pending: commit.
