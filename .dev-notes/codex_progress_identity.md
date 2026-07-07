# codex progress: identity lane

## Decisions
- Progress file name follows lane rule: `.dev-notes/codex_progress_identity.md`.
- ResultStore vintage data identity will be content-derived from enumerable vintage labels, reference-calendar summary, and a bounded last-resolved-vintage panel fingerprint. Callable-only vintage sources without enumerable vintages are undigestible for disk result-store reuse unless the source carries `__mf_digest__`.
- ResultStore digest will include effective selection seed resolved from `macroforecast.meta.get_config()["random_seed"]`, matching the runner, plus arm-relevant backend package versions. `macroforecast.__version__` stays outside the digest and remains a reuse-time warning.
- Rescore identity will reuse the result-store cell identity payload/digest machinery where possible, written as a small checkpoint cell manifest at pipeline execution time. Legacy checkpoint cells without this manifest warn rather than fail.
- Preprocessing disk-cache callable handling will distinguish disk-safe specs from in-memory use: custom preprocessing callables need `__mf_digest__` for disk tiers; lambda custom steps are rejected when constructing a spec because they cannot be safely keyed for disk reuse.

## Gate Log
- `~/project/macroforecast/.venv/bin/python -m compileall -q macroforecast/pipeline/result_store.py macroforecast/pipeline/run.py macroforecast/pipeline/rescore.py macroforecast/preprocessing/cache.py macroforecast/preprocessing/specs.py macroforecast/preprocessing/preprocess.py macroforecast/forecasting/preprocessing_stage.py tests/pipeline/test_result_store.py tests/pipeline/test_rescore.py tests/pipeline/test_preprocessing_share.py tests/preprocessing/test_fitted_cache.py tests/preprocessing/test_preprocess.py` — passed.
- `~/project/macroforecast/.venv/bin/python -m pytest tests/preprocessing/test_fitted_cache.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider` — passed, 8 tests.
- `~/project/macroforecast/.venv/bin/python -m pytest tests/preprocessing/test_preprocess.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider` — passed, 33 tests.
- `~/project/macroforecast/.venv/bin/python -m pytest tests/pipeline/test_result_store.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider` — passed, 10 tests.
- `~/project/macroforecast/.venv/bin/python -m pytest tests/pipeline/test_rescore.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider` — passed, 9 tests; existing subsample-size warnings only.
- `~/project/macroforecast/.venv/bin/python -m pytest tests/pipeline/test_preprocessing_share.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider` — passed, 10 tests; multiprocessing fork deprecation warnings only.
- `~/project/macroforecast/.venv/bin/python -m pytest tests/pipeline/test_preprocessing_share.py::test_disk_store_custom_callable_requires_digest_for_reuse --timeout=300 --timeout-method=thread -q -p no:cacheprovider` — passed after capturing the expected undigestible-callable warnings.
- `~/project/macroforecast/.venv/bin/python -m tools.docgen docs/reference` — regenerated reference docs.
- `~/project/macroforecast/.venv/bin/python -m tools.docgen --check docs/reference` — passed.
- `~/project/macroforecast/.venv/bin/python -m pytest tests/pipeline --timeout=300 --timeout-method=thread -q -p no:cacheprovider` — passed, 227 tests in 849.13s; warnings were existing ragged-coverage, multiprocessing fork deprecation, statsmodels frequency, rescore legacy-checkpoint, and small-subsample diagnostics.
- `~/project/macroforecast/.venv/bin/python -m pytest tests/forecasting --timeout=300 --timeout-method=thread -q -p no:cacheprovider` — passed, 160 tests in 393.59s; warnings were existing default-feature, statsmodels convergence/frequency, unsupported-direct-policy, and vintage target-transform diagnostics.
- `~/project/macroforecast/.venv/bin/python -m mypy macroforecast` — passed, no issues in 109 source files.
- `~/project/macroforecast/.venv/bin/python -m tools.docgen --check docs/reference` — passed after all edits.
- `git diff --check` — passed.

## Pending Gates
- None.
