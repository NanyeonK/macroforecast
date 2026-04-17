# Macrocast 2026-04-15 Checkpoint + CW Implementation Plan

> Worker note: execute in order. Keep the docs/API checkpoint separate from the CW implementation change.

Goal
- Safely checkpoint the already-tested public-surface/docs refactor on server1, then extend the evaluation layer by making `stat_test = cw` operational with compiler/runtime/docs/test coverage.

Architecture
- First freeze the current public-surface expansion as its own clean checkpoint so the working tree matches the tested state already present on server1.
- Then implement CW as the next narrow evaluation slice on top of the existing benchmark-aware execution path, reusing compiler provenance, manifest writing, and prediction error outputs already used by DM.

Tech stack
- Python 3.12
- pytest
- pandas / numpy / scikit-learn / statsmodels
- git

Work scope
- In scope:
  - review and commit the current docs/API export refactor
  - promote `cw` from planned to operational in registry truth
  - add compiler/runtime/test/docs support for CW artifacts
  - run regression tests and a real smoke execution
- Out of scope:
  - `mcs`
  - SHAP
  - new benchmark-family/plugin execution
  - preprocessing widening beyond current slice

Verification strategy
- Level: test-suite
- Primary command: `python3 -m pytest tests/test_stage0.py tests/test_stage0_completion.py tests/test_raw.py tests/test_raw_adapters.py tests/test_raw_hardened.py tests/test_raw_sd.py tests/test_recipe_execution.py tests/test_preprocess_contract.py tests/test_execution_pipeline.py tests/test_compiler.py -q`
- Focused CW commands:
  - `python3 -m pytest tests/test_execution_pipeline.py tests/test_compiler.py -q`
- Smoke command:
  - `python3 -m pytest tests/test_execution_pipeline.py::test_execute_recipe_writes_cw_artifact -q`
- What this validates:
  - current public package surfaces still import and test cleanly
  - CW compiles as executable, emits a dedicated artifact, and records provenance without regressing existing execution slices

## Task 1: Checkpoint the current public-surface/docs refactor

Dependencies: None
Files:
- Modify: `docs/index.md`
- Modify: `docs/api/index.md`
- Create: `docs/compiler.md`
- Create: `docs/execution.md`
- Create: `docs/preprocessing.md`
- Create: `docs/registry.md`
- Create: `docs/api/compiler.md`
- Create: `docs/api/execution.md`
- Create: `docs/api/preprocessing.md`
- Create: `docs/api/registry.md`
- Modify: `macrocast/__init__.py`
- Modify: `macrocast/recipes/build.py`
- Modify: `macrocast/recipes/construct.py`
- Modify: `macrocast/recipes/types.py`
- Create: `macrocast/compiler/*`
- Create: `macrocast/execution/*`
- Create: `macrocast/preprocessing/*`
- Create: `macrocast/registry/*`
- Create: `examples/recipes/*`

- [ ] Review the current diff with `git diff --stat` and `git diff -- <file>` for every tracked modified file.
- [ ] Run the primary regression command before committing. Expected: all pass.
- [ ] Run a top-level import smoke check:
  - `python3 - <<'PY'`
  - `import macrocast`
  - `assert hasattr(macrocast, "compile_recipe_dict")`
  - `assert hasattr(macrocast, "execute_recipe")`
  - `assert hasattr(macrocast, "build_preprocess_contract")`
  - `PY`
- [ ] Stage only the intended refactor files listed above.
- [ ] Commit with message: `feat: expose compiler execution preprocessing and registry surfaces`

## Task 2: Add failing CW tests first

Dependencies: Task 1
Files:
- Modify: `tests/test_execution_pipeline.py`
- Modify: `tests/test_compiler.py`

- [ ] Add a runtime test that requests `stat_test = cw`, runs `execute_recipe(...)`, and asserts:
  - manifest contains `stat_test_spec.stat_test == "cw"`
  - manifest contains `stat_test_file == "stat_test_cw.json"`
  - artifact file `stat_test_cw.json` exists
  - artifact payload includes `stat_test == "cw"`
  - artifact payload includes `forecast_adjustment_mean`
- [ ] Add a compiler/governance test that asserts `cw` is operational in `axis_governance_table()`.
- [ ] Add a compile-and-run test that compiles a recipe with `6_stat_tests.stat_test = cw`, runs it, and asserts the manifest records `stat_test_cw.json`.
- [ ] Run `python3 -m pytest tests/test_execution_pipeline.py tests/test_compiler.py -q` and confirm RED failure because CW is not operational yet.

## Task 3: Implement CW operational support

Dependencies: Task 2
Files:
- Modify: `macrocast/registry/build.py`
- Modify: `macrocast/execution/build.py`

- [ ] Promote `stat_test.cw` from `planned` to `operational` in the axis registry.
- [ ] In execution runtime, add a dedicated CW computation helper using the existing prediction table:
  - use benchmark and model squared errors plus benchmark-vs-model forecast difference
  - compute a mean adjusted differential and a simple standard-error-based test statistic
  - fail explicitly when fewer than two forecast errors are available or the variance estimate is non-positive
- [ ] Write a dedicated `stat_test_cw.json` artifact when `stat_test_spec.stat_test == "cw"`.
- [ ] Update manifest writing so `stat_test_file` records `stat_test_cw.json` for CW, preserving current DM behavior.
- [ ] Re-run `python3 -m pytest tests/test_execution_pipeline.py tests/test_compiler.py -q` and confirm GREEN.

## Task 4: Update public docs and roadmap status for CW

Dependencies: Task 3
Files:
- Modify: `docs/index.md`
- Modify: `docs/api/index.md`
- Modify: `docs/compiler.md`
- Modify: `docs/execution.md`
- Modify: `plans/plan_04_14_1958.md`

- [ ] Update docs so current operational statistical tests list `dm` and `cw`.
- [ ] In compiler docs, state that CW is executable and produces a dedicated runtime artifact.
- [ ] In execution docs, document the CW artifact filename and first-slice semantics honestly.
- [ ] In roadmap status, move Phase E wording from DM-only to DM+CW operational.

## Task 5: Final verification and smoke run

Dependencies: Task 4
Files:
- Read-only verification

- [ ] Run `python3 -m pytest tests/test_execution_pipeline.py tests/test_compiler.py -q`. Expected: pass.
- [ ] Run the primary regression command. Expected: pass.
- [ ] Run a direct smoke execution that compiles or executes a CW recipe on `tests/fixtures/fred_md_ar_sample.csv` and verify:
  - `stat_test_cw.json` is written
  - manifest points to the file
  - existing predictions/metrics artifacts still exist
- [ ] Commit CW implementation and doc updates with message: `feat: operationalize clark-west evaluation artifact`
