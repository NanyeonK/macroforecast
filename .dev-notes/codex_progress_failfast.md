# Lane progress: failfast

## Decisions
- Progress file name: `.dev-notes/codex_progress_failfast.md`, derived from worktree suffix `mf-failfast`.
- Scope: keep edits inside the worktree and the files owned by the failfast workplan; leave `.dev-notes/codex_progress.md` untouched because it is not lane-specific.
- `pipeline_spec()` resolves string arm models to `ModelSpec` objects during spec construction and rejects bare callables with the `custom_model()` pointer.
- `custom_model()` accepts `default_preset=None` only when no search spaces are supplied; with search spaces it must name an available preset.
- Recursive custom-supervised exogenous-feature incompatibility is emitted as a spec-build warning because it is detectable from resolved target policy + custom model + feature spec.

## Gate log
- Quick syntax check: `~/project/macroforecast/.venv/bin/python -m py_compile macroforecast/models/specs.py macroforecast/pipeline/spec.py macroforecast/pipeline/run.py macroforecast/pipeline/evaluate.py macroforecast/reporting/core.py` -> pass.
- Blanket-except check: `rg -n "except Exception|except:" macroforecast/pipeline/evaluate.py` -> pass, no matches.
- New model tests: `~/project/macroforecast/.venv/bin/python -m pytest tests/models/test_custom_model_failfast.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider` -> pass (5 passed).
- New pipeline tests: `~/project/macroforecast/.venv/bin/python -m pytest tests/pipeline/test_failfast_validation.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider` -> pass after fixing a test import shadowing issue (12 passed).
- Focused evalspec rerun: `~/project/macroforecast/.venv/bin/python -m pytest tests/pipeline/test_evalspec_threading.py tests/pipeline/test_failfast_validation.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider` -> pass (45 passed).
- Pipeline directory gate: `~/project/macroforecast/.venv/bin/python -m pytest tests/pipeline --timeout=300 --timeout-method=thread -q -p no:cacheprovider` -> initial run failed because an existing unknown-metric test expected evaluate-time failure; updated to spec-build failure per workplan; rerun passed (232 passed, 57 warnings).
- Models directory gate: `~/project/macroforecast/.venv/bin/python -m pytest tests/models --timeout=300 --timeout-method=thread -q -p no:cacheprovider` -> pass after adding the volatility custom-model constructor contract test (180 passed, 9 warnings).
- Reporting affected-surface check: `~/project/macroforecast/.venv/bin/python -m pytest tests/reporting --timeout=300 --timeout-method=thread -q -p no:cacheprovider` -> pass (15 passed).
- Docgen: `~/project/macroforecast/.venv/bin/python -m tools.docgen docs/reference` -> pass, wrote 37 reference pages.
- Docgen drift check: `~/project/macroforecast/.venv/bin/python -m tools.docgen --check docs/reference` -> pass after final docstring update.
- mypy: `~/project/macroforecast/.venv/bin/python -m mypy macroforecast` -> pass after final code update (`Success: no issues found in 109 source files`).
- Result-store warning cleanup check: `~/project/macroforecast/.venv/bin/python -m pytest tests/pipeline/test_result_store.py::test_result_store_custom_callable_requires_digest_opt_in --timeout=300 --timeout-method=thread -q -p no:cacheprovider` -> pass (1 passed, 2 expected warnings).
- Final focused failfast rerun: `~/project/macroforecast/.venv/bin/python -m pytest tests/models/test_custom_model_failfast.py tests/pipeline/test_failfast_validation.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider` -> pass (17 passed).
- Final whitespace check: `git diff --check` -> pass.
- Final blanket-except check: `rg -n "except Exception|except:" macroforecast/pipeline/evaluate.py` -> pass, no matches.
