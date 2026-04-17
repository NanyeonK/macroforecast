# Macrocast 2026-04-15 Phase B Preprocessing Widening Plan

Goal
- Start Phase B by promoting one additional preprocessing path from registry-only to operational while keeping the current comparison/stat-test/benchmark artifact layer fixed.

Architecture
- Keep the current preprocessing language unchanged and widen only one narrow raw-panel path.
- Promote `scaling_policy = robust` on the existing `extra_preprocess_without_tcode` + `x_missing_policy = em_impute` + `preprocess_fit_scope = train_only` route so the package gains one additional operational preprocessing variant without adding new transform families.

Tech stack
- Python 3.12
- pytest
- scikit-learn preprocessing utilities
- git

Work scope
- In scope:
  - TDD for one new operational preprocessing contract
  - runtime support for robust scaling on the current raw-panel path
  - registry/docs/roadmap updates
  - focused and broader regression plus smoke verification
- Out of scope:
  - tcode execution paths
  - transformed-scale evaluation
  - feature selection / PCA widening
  - multi-axis preprocessing sweep execution

Verification strategy
- Level: test-suite
- Focused command: `python3 -m pytest tests/test_preprocess_contract.py tests/test_execution_pipeline.py tests/test_compiler.py -q`
- Broader command: `python3 -m pytest tests/test_stage0.py tests/test_stage0_completion.py tests/test_raw.py tests/test_raw_adapters.py tests/test_raw_hardened.py tests/test_raw_sd.py tests/test_recipe_execution.py tests/test_preprocess_contract.py tests/test_execution_pipeline.py tests/test_compiler.py -q`
- What this validates:
  - robust-scaling train-only raw-panel contracts compile and execute
  - manifest/provenance remain stable
  - current comparison/stat-test/custom-benchmark slices do not regress

## Task 1: Add failing robust-scaling preprocessing tests

Dependencies: None
Files:
- Modify: `tests/test_preprocess_contract.py`
- Modify: `tests/test_execution_pipeline.py`
- Modify: `tests/test_compiler.py`

- [ ] Add a preprocessing contract test asserting the robust-scaling train-only raw-panel contract is operational.
- [ ] Add a runtime execution test asserting a recipe with `extra_preprocess_without_tcode + x_missing_policy=em_impute + scaling_policy=robust + preprocess_fit_scope=train_only` executes and preserves the preprocess contract in the manifest.
- [ ] Add a compiler test asserting such a recipe compiles as executable and can run successfully.
- [ ] Run the focused command and confirm RED.

## Task 2: Implement robust-scaling operational support

Dependencies: Task 1
Files:
- Modify: `macrocast/preprocessing/build.py`
- Modify: `macrocast/registry/build.py`
- Modify: `macrocast/execution/build.py`

- [ ] Promote `scaling_policy.robust` from registry-only to operational.
- [ ] Extend `is_operational_preprocess_contract()` so it recognizes the robust-scaling train-only raw-panel contract in addition to the existing standard-scaling path.
- [ ] Extend raw-panel runtime preprocessing to apply `RobustScaler` when `scaling_policy == "robust"`.
- [ ] Keep fit behavior train-only and fail-closed for unsupported preprocessing requests.
- [ ] Re-run the focused command and confirm GREEN.

## Task 3: Update docs and roadmap

Dependencies: Task 2
Files:
- Modify: `docs/index.md`
- Modify: `docs/api/index.md`
- Modify: `docs/preprocessing.md`
- Modify: `docs/compiler.md`
- Modify: `docs/execution.md`
- Modify: `plans/plan_04_14_1958.md`

- [ ] Update docs so the operational preprocessing subset now includes both the standard-scaling and robust-scaling train-only raw-panel path.
- [ ] Keep wording honest that preprocessing widening is still narrow and not yet a general sweep engine.
- [ ] Update roadmap wording so Phase B reflects one additional preprocessing variant as complete for the current slice.

## Task 4: Final verification and checkpoint

Dependencies: Task 3
Files:
- Read-only verification

- [ ] Run the focused command. Expected: pass.
- [ ] Run the broader command. Expected: pass.
- [ ] Run a smoke execution using the robust-scaling path and verify predictions, metrics, comparison summary, and manifest preprocess provenance all exist.
- [ ] Commit with message: `feat: widen preprocessing with robust scaling path`
