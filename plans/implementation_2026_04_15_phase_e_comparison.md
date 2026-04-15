# Macrocast 2026-04-15 Phase E Comparison Artifact Plan

Goal
- Extend Phase E beyond single-test artifacts by adding a persistent comparison summary artifact that packages model-vs-benchmark evaluation results in one benchmark-aware file.

Architecture
- Keep predictions.csv as the row-level truth and keep DM/CW artifacts as optional test-specific outputs.
- Add a deterministic `comparison_summary.json` built directly from the existing prediction table so every executable run emits one stable comparison-layer artifact even when no statistical test is requested.

Tech stack
- Python 3.12
- pytest
- pandas / numpy
- git

Work scope
- In scope:
  - TDD for comparison summary artifact generation
  - runtime manifest linkage for the new artifact
  - docs and roadmap updates
  - focused and broader regression plus smoke verification
- Out of scope:
  - new statistical tests
  - multi-model comparison bundles
  - SHAP / importance widening
  - benchmark plugin changes

Verification strategy
- Level: test-suite
- Focused command: `python3 -m pytest tests/test_execution_pipeline.py tests/test_compiler.py -q`
- Broader command: `python3 -m pytest tests/test_stage0.py tests/test_stage0_completion.py tests/test_raw.py tests/test_raw_adapters.py tests/test_raw_hardened.py tests/test_raw_sd.py tests/test_recipe_execution.py tests/test_preprocess_contract.py tests/test_execution_pipeline.py tests/test_compiler.py -q`
- What this validates:
  - every executable run emits a stable comparison summary artifact
  - manifest links the artifact cleanly
  - current DM/CW/custom-benchmark slices remain intact

## Task 1: Add failing comparison-artifact tests

Dependencies: None
Files:
- Modify: `tests/test_execution_pipeline.py`
- Modify: `tests/test_compiler.py`

- [ ] Add a runtime test that executes a normal run with no stat test and asserts:
  - `comparison_summary.json` exists
  - manifest records `comparison_file == "comparison_summary.json"`
  - comparison payload preserves `benchmark_name`, `model_name`, and horizon summaries
- [ ] Add a test asserting comparison payload includes horizon-level fields such as `n_predictions`, `mean_loss_diff`, `win_rate`, `model_msfe`, and `benchmark_msfe`.
- [ ] Add a compile-and-run test asserting compiled execution also writes and links the comparison artifact.
- [ ] Run the focused command and confirm RED.

## Task 2: Implement comparison summary artifact

Dependencies: Task 1
Files:
- Modify: `macrocast/execution/build.py`

- [ ] Add a helper that derives a benchmark-aware comparison summary from `predictions.csv` data before optional DM/CW/importance steps.
- [ ] Write `comparison_summary.json` for every successful run.
- [ ] Record `comparison_file` in manifest provenance.
- [ ] Keep the artifact independent of optional stat tests so it always exists.
- [ ] Re-run the focused command and confirm GREEN.

## Task 3: Update docs and roadmap

Dependencies: Task 2
Files:
- Modify: `docs/index.md`
- Modify: `docs/api/index.md`
- Modify: `docs/compiler.md`
- Modify: `docs/execution.md`
- Modify: `plans/plan_04_14_1958.md`

- [ ] Document `comparison_summary.json` as the current baseline comparison-layer artifact.
- [ ] Clarify that DM/CW remain optional statistical-test artifacts on top of the always-written comparison summary.
- [ ] Update roadmap wording so Phase E now includes both stat-test support and the baseline comparison artifact layer.

## Task 4: Final verification and checkpoint

Dependencies: Task 3
Files:
- Read-only verification

- [ ] Run the focused command. Expected: pass.
- [ ] Run the broader command. Expected: pass.
- [ ] Run a smoke execution and verify `comparison_summary.json`, `metrics.json`, and `predictions.csv` all exist and the manifest links the comparison artifact.
- [ ] Commit with message: `feat: add comparison summary artifact`
