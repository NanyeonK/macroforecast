# Macrocast 2026-04-15 Wrapper Handoff Contract Plan

Goal
- Define the smallest honest wrapper/orchestrator handoff contract for wrapper-owned families without pretending they are executable single-run paths.

Architecture
- Keep single-run execution exactly as it is.
- For `study_mode='orchestrated_bundle_study'`, compiler should emit a deterministic wrapper handoff payload that tells a future wrapper what family of bundle is requested and what shared study environment is already locked.

Tech stack
- Python 3.12
- pytest
- existing Stage 0 / compiler manifest machinery
- git

Work scope
- In scope:
  - TDD for wrapper handoff contract emission
  - compiler-side validation for minimal wrapper metadata
  - manifest serialization of `wrapper_handoff`
  - docs and roadmap updates
- Out of scope:
  - actual wrapper execution
  - benchmark-suite runner implementation
  - ablation orchestration
  - multi-target heterogeneous bundle execution

Verification strategy
- Level: test-suite
- Focused command: `python3 -m pytest tests/test_compiler.py tests/test_stage0_completion.py -q`
- Broader command: `python3 -m pytest tests/test_stage0.py tests/test_stage0_completion.py tests/test_raw.py tests/test_raw_adapters.py tests/test_raw_hardened.py tests/test_raw_sd.py tests/test_recipe_execution.py tests/test_preprocess_contract.py tests/test_execution_pipeline.py tests/test_compiler.py -q`
- What this validates:
  - wrapper-owned studies compile into deterministic non-executable handoff manifests
  - missing wrapper metadata fails closed
  - existing single-run executable slices do not regress

## Task 1: Add failing wrapper-handoff tests

Dependencies: None
Files:
- Modify: `tests/test_compiler.py`
- Modify: `tests/test_stage0_completion.py`

- [ ] Add a compiler test for `study_mode='orchestrated_bundle_study'` with valid wrapper metadata and assert:
  - compile succeeds
  - execution status is `representable_but_not_executable`
  - manifest contains `wrapper_handoff`
  - handoff payload contains `wrapper_family`, `bundle_label`, `route_owner`, and `execution_posture`
- [ ] Add a failure test asserting wrapper studies without required wrapper metadata fail closed.
- [ ] Add a Stage 0 route-owner test asserting orchestrated bundle studies still resolve to `wrapper`.
- [ ] Run the focused command and confirm RED.

## Task 2: Implement minimal wrapper handoff contract

Dependencies: Task 1
Files:
- Modify: `macrocast/compiler/build.py`
- Modify: `macrocast/compiler/types.py`

- [ ] Require minimal wrapper metadata in `leaf_config` when `study_mode='orchestrated_bundle_study'`:
  - `wrapper_family`
  - `bundle_label`
- [ ] Accept the first explicit wrapper families:
  - `multi_target_separate_runs`
  - `benchmark_suite`
  - `ablation_study`
- [ ] Add compiler helper that builds a deterministic `wrapper_handoff` payload from Stage 0 + recipe context.
- [ ] Include `wrapper_handoff` in compiled manifest serialization only for wrapper-owned routes.
- [ ] Keep execution status non-executable for wrapper-owned routes; this is a handoff contract, not a hidden runner.
- [ ] Re-run the focused command and confirm GREEN.

## Task 3: Update docs and roadmap

Dependencies: Task 2
Files:
- Modify: `docs/stage0.md`
- Modify: `docs/compiler.md`
- Modify: `docs/index.md`
- Modify: `plans/plan_04_14_1958.md`

- [ ] Document wrapper-owned studies as handoff contracts, not executable single-run paths.
- [ ] Document the first wrapper families and required metadata fields.
- [ ] Update roadmap so post-Phase-G architecture now has a minimal wrapper handoff contract defined.

## Task 4: Final verification and checkpoint

Dependencies: Task 3
Files:
- Read-only verification

- [ ] Run the focused command. Expected: pass.
- [ ] Run the broader command. Expected: pass.
- [ ] Run one compile smoke for an orchestrated bundle study and verify `wrapper_handoff` in the manifest.
- [ ] Commit with message: `feat: add wrapper handoff contract`
