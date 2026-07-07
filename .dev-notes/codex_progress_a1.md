# A1 hist_mean lane progress

## Decisions
- 2026-07-08: Workplan file references `macroforecast/models/persistence.py` for persistence models, but current code keeps the target-only naive/random-walk implementations in `macroforecast/models/timeseries.py`; use the current location and note this as code movement, not a design deviation.
- 2026-07-08: Implement `hist_mean` as one registry model with `input_kind="target"` and parameter `window: int | None = None`; no pipeline wrapper or policy-specific special case.
- 2026-07-08: Keep `hist_mean` out of `DIRECT_POLICY_GUARD_MODELS` and `DIRECT_AVERAGE_GUARD_MODELS`. It has no iterated dynamics, and the direct/direct-average runner hands it the already transformed forecast object, so its horizon-invariant mean is a valid direct-safe constant projection.
- 2026-07-08: Do not add name-based CW special-casing. The evaluator licenses CW through `Arm(nested_in_benchmark=True)` for the larger contender; the smoke test will mark AR nested against the `hist_mean` benchmark, which preserves the existing public contract.
- 2026-07-08: `hist_mean` missing-target behavior follows the existing target-only baseline convention: if no finite target values are available after fit-window filtering, the fitted constant is `0.0`.

## S1 mechanism notes
- `ModelSpec.input_kind="target"` dispatches `ModelSpec.fit(X, y)` to `fit_func(target, **params)` using `y` when present, so a target-only model should ignore features naturally.
- Target-only naive models return a `ModelFit` built by `fit_estimator` over a dummy one-column feature frame; their `predict(X)` length drives the target-only path length.
- The policy matrix is generated from `mf.list_model_specs()` plus `DIRECT_POLICY_GUARD_MODELS` and `DIRECT_AVERAGE_GUARD_MODELS`; no separate matrix metadata file needs editing beyond regeneration.
- Clark-West rows are emitted only for contenders whose arm declares `nested_in_benchmark=True` and when `EvalSpec.tests` includes `"cw"` with the default `cw_for_nested=True`.

## Gates
- PASS: `~/project/macroforecast/.venv/bin/python -m pytest tests/models/test_hist_mean.py tests/pipeline/test_hist_mean_pipeline.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider` (9 passed, 2.60s).
- PASS: `~/project/macroforecast/.venv/bin/python -m pytest tests/pipeline/test_direct_policy_guard.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider` (21 passed, 6 warnings from existing statsmodels frequency inference, 1.71s).
- PASS: `~/project/macroforecast/.venv/bin/python tools/gen_model_overview.py --out docs/guide` (wrote 13 pages).
- PASS: `~/project/macroforecast/.venv/bin/python tools/gen_policy_matrix.py --out docs/guide` (wrote 1 page).
- PASS: `~/project/macroforecast/.venv/bin/python -m tools.docgen docs/reference` (wrote 37 pages).
- PASS: `~/project/macroforecast/.venv/bin/python -m tools.docgen --check docs/reference` (`docs/reference` up to date).
- PASS: `~/project/macroforecast/.venv/bin/python tools/gen_model_overview.py --check docs/guide` (13 model pages in sync).
- PASS: `~/project/macroforecast/.venv/bin/python tools/gen_policy_matrix.py --check docs/guide` (1 policy matrix page in sync).
- PASS: `~/project/macroforecast/.venv/bin/python -m pytest tests/models --timeout=300 --timeout-method=thread -q -p no:cacheprovider` (188 passed, 9 warnings from existing statsmodels/BVAR/torch paths, 174.61s).
- PASS: `~/project/macroforecast/.venv/bin/python -m pytest tests/pipeline --timeout=300 --timeout-method=thread -q -p no:cacheprovider` (249 passed, 62 warnings from existing ragged-sample, multiprocessing, rescore, statsmodels, result-store, and intentional failed-cell paths, 830.49s).
- PASS: `~/project/macroforecast/.venv/bin/python -m mypy macroforecast` (no issues in 111 source files).
- PASS: `git diff --check` (no whitespace errors).
- PASS: local commit created with message `feat(models): add historical mean benchmark`.
