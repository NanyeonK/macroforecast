# Macrocast 2026-04-15 Phase F Importance Widening Plan

Goal
- Continue Phase F by widening `minimal_importance` to one additional supported route before any SHAP work.

Architecture
- Keep the current importance design unchanged: `minimal_importance` remains a narrow first-slice artifact built from the final fitted training window.
- Extend it to one additional non-AR linear route, `lasso` on `raw_feature_panel`, while keeping current preprocessing/comparison/stat-test/benchmark behavior fixed.

Tech stack
- Python 3.12
- pytest
- numpy / pandas / scikit-learn
- git

Work scope
- In scope:
  - TDD for `minimal_importance` on `lasso + raw_feature_panel`
  - runtime widening in the importance layer
  - docs and roadmap updates
  - focused and broader regression plus smoke verification
- Out of scope:
  - SHAP
  - lagged-target importance semantics
  - broad multi-model importance framework redesign
  - benchmark or preprocessing widening beyond current fixed slices

Verification strategy
- Level: test-suite
- Focused command: `python3 -m pytest tests/test_execution_pipeline.py tests/test_compiler.py -q`
- Broader command: `python3 -m pytest tests/test_stage0.py tests/test_stage0_completion.py tests/test_raw.py tests/test_raw_adapters.py tests/test_raw_hardened.py tests/test_raw_sd.py tests/test_recipe_execution.py tests/test_preprocess_contract.py tests/test_execution_pipeline.py tests/test_compiler.py -q`
- What this validates:
  - minimal importance works on one new supported route
  - manifest/artifact contracts remain stable
  - current custom benchmark, comparison, preprocessing, and CW/DM slices do not regress

## Task 1: Add failing lasso-importance tests

Dependencies: None
Files:
- Modify: `tests/test_execution_pipeline.py`
- Modify: `tests/test_compiler.py`

- [ ] Add a runtime test that requests `importance_method = minimal_importance` for `model_family = lasso` and `feature_builder = raw_feature_panel`.
- [ ] Assert the run writes `importance_minimal.json` and that the payload records `model_family == "lasso"` with non-empty `feature_importance`.
- [ ] Add a compile-and-run test asserting a lasso raw-panel recipe with minimal importance is executable and writes the artifact.
- [ ] Run the focused command and confirm RED.

## Task 2: Implement lasso minimal-importance support

Dependencies: Task 1
Files:
- Modify: `macrocast/execution/build.py`
- Modify: `docs/execution.md`
- Modify: `docs/compiler.md`
- Modify: `docs/index.md`
- Modify: `docs/api/index.md`
- Modify: `plans/plan_04_14_1958.md`

- [ ] Extend `_compute_minimal_importance()` so `lasso` is supported on `raw_feature_panel`.
- [ ] Reuse the current final-window coefficient-magnitude semantics already used for `ridge`.
- [ ] Keep fail-closed behavior for unsupported model families and unsupported feature builders.
- [ ] Update docs and roadmap so the first operational importance slice now covers `ridge`, `lasso`, and `randomforest`.
- [ ] Re-run the focused command and confirm GREEN.

## Task 3: Final verification and checkpoint

Dependencies: Task 2
Files:
- Read-only verification

- [ ] Run the focused command. Expected: pass.
- [ ] Run the broader command. Expected: pass.
- [ ] Run a smoke execution for lasso minimal importance and verify `importance_minimal.json`, `comparison_summary.json`, and manifest importance provenance all exist.
- [ ] Commit with message: `feat: widen minimal importance to lasso route`
