# Runtime Lane Progress

Worktree: `/home/nanyeon99/project/mf-runtime`
Branch target: `fix/runtime-seeds-parallel`

## Decisions

- Progress file name: using `.dev-notes/codex_progress_runtime.md` from the `mf-runtime` worktree suffix, per lane ground rules.
- Scope: keep edits inside runtime lane owned regions unless a test/doc requires a narrow adjacent change.
- Seed design: follow the workplan's fixed design: `PipelineSpec.seed` becomes the run-scoped meta seed, workers re-apply it, per-arm `random_state` is derived only when the user did not explicitly provide one, and provenance records the effective seed set.
- Parallel design: combine BLAS env setup, `n_jobs`, seed re-apply, and worker data install in a single worker initializer.
- Model-store default: flip only the pipeline default to `save_models=False`; keep the low-level `forecasting.run(..., save_models=True)` default for backward compatibility.
- Basic provenance compatibility: `effective_seeds` is recorded only at full provenance level so `provenance_level="basic"` keeps its old top-level shape.
- Vintage measurement: local legacy-style full-spec-per-task dispatch vs current data-once initializer on a mapping-backed synthetic vintage run (10 origins, 2 arms, 4 horizons, `n_jobs=2`, 80 extra columns) was 14.7659s vs 14.5305s, a 1.02x speedup. A 1-horizon/2-cell variant measured 0.6066s vs 0.6206s because each worker received only one task, so the per-worker cache could not amortize.

## Deviations

- The workplan's measurement shape said 10 origins x 2 arms; with one horizon that creates only two parallel cells, so there is no repeated per-worker vintage-source reuse to measure. I also recorded a 4-horizon variant to exercise the repaired repeated-cell path.

## Gates

- `~/project/macroforecast/.venv/bin/python -m pytest tests/pipeline/test_runtime_integrity.py tests/pipeline/test_auto_parallelism.py tests/pipeline/test_vintage_pipeline.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider` — PASS (27 passed; fork deprecation warnings only).
- `~/project/macroforecast/.venv/bin/python -m tools.docgen docs/reference` — PASS (wrote 37 pages).
- `~/project/macroforecast/.venv/bin/python -m pytest tests/pipeline --timeout=300 --timeout-method=thread -q -p no:cacheprovider` — FAIL first run (227 passed, 2 failed): `provenance_level="basic"` included new `effective_seeds`; `test_spec_stage0` expected old `save_models=True` default.
- `~/project/macroforecast/.venv/bin/python -m pytest tests/pipeline/test_default_provenance.py::test_provenance_level_basic_matches_pre_change_shape tests/pipeline/test_spec_stage0.py::test_pipeline_spec_builds_and_resolves tests/pipeline/test_runtime_integrity.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider` — PASS (9 passed; fork deprecation warnings only).
- `~/project/macroforecast/.venv/bin/python -m pytest tests/pipeline --timeout=300 --timeout-method=thread -q -p no:cacheprovider` — PASS (229 passed; known warnings only).
- `~/project/macroforecast/.venv/bin/python -m pytest tests/forecasting --timeout=300 --timeout-method=thread -q -p no:cacheprovider` — PASS (160 passed; known warnings only).
- `~/project/macroforecast/.venv/bin/python -m mypy macroforecast` — PASS (no issues in 110 source files).
- `~/project/macroforecast/.venv/bin/python -m tools.docgen --check docs/reference` — PASS.
- `~/project/macroforecast/.venv/bin/python -m pytest tests/pipeline/test_runtime_integrity.py::test_parallel_worker_initializer_sets_threads_seed_and_data_once --timeout=300 --timeout-method=thread -q -p no:cacheprovider` — PASS (1 passed; rerun after changing worker BLAS env caps from `setdefault` to assignment).
- Final rerun after env-cap assignment: `~/project/macroforecast/.venv/bin/python -m mypy macroforecast` — PASS; `~/project/macroforecast/.venv/bin/python -m tools.docgen --check docs/reference` — PASS.
