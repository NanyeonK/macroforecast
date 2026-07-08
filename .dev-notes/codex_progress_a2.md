# A2 splitter/IC lane progress

## S1 investigation note

- Read `.dev-notes/lane_ground_rules.md` and `.dev-notes/workplan_a2_splitter_ic_codex.md`.
- The referenced Phase B oracle copies named in the workplan are not present under `.dev-notes/` beyond the workplan text itself; searched `.dev-notes`, docs, tests, `macroforecast/model_selection`, `macroforecast/forecasting`, and `macroforecast/pipeline` for Hounyo/GCLS/Han keywords.
- Current validation split construction:
  - `macroforecast.window.WindowSpec` owns outer estimation/test windows, retune cadence, and per-origin validation folds.
  - `forecasting/runner.py` places absolute `item["val_splits"]` on each origin and remaps them to feature-matrix-relative folds with `_relative_splits_for_index`.
  - `forecasting/selection_stage.py::_availability_safe_selection_splits` rebuilds availability-safe folds for direct/recursive policies from the `WindowSpec` or from absolute folds.
  - `model_selection.search.select_params` already accepts explicit `splits=` and records `"split_source": "explicit"`.
- Grid semantics:
  - `model_selection.builders._candidates` already evaluates `itertools.product` over all `SearchSpec.param_grid` keys for `fixed`, `grid`, and `cv_path`.
  - Joint grid behavior therefore exists, but it is not documented or pinned by a test that proves two parameters are scored together.
- IC computation:
  - `model_selection.search.select_by_information_criterion` already fits each candidate on the full supplied sample and scores Gaussian RSS-based AIC/BIC/AICc from `ssr_`, `nobs_`, and `n_params_`.
  - AR/FAR already expose `selection_method="bic"` and route through this helper in `forecasting/policies/base.py`.
  - The helper currently catches `Exception` while recording candidate failures; this lane will replace that with explicit failure classes to satisfy the no-blanket-except ground rule for edited code.
  - IC support boundary remains models whose fitted estimator exposes `ssr_`, `nobs_`, and `n_params_`; non-supporting models should raise an actionable `SearchError`/`ValueError`.
- Store identity:
  - `pipeline/result_store.py::result_cell_identity` already includes `arm.model_selection` via `_object_identity`.
  - `SearchSpec.to_dict()` feeds that identity, so adding splitter/criterion fields there should make fold-boundary and IC-route changes digest-sensitive.
  - Callable splitters must be blocked from digest reuse unless they carry `__mf_digest__`, reusing the result-store callable-digest rule.

## Decisions

- Add a model-selection validation splitter spec in `macroforecast/model_selection/types.py` rather than changing `WindowSpec` shape. This keeps the splitter composable per `SearchSpec` and avoids perturbing existing `poos`/`kfold` window behavior.
- Implement explicit splitter generation in model-selection code and have forecasting selection-stage call the same resolver when a selected `SearchSpec` carries an override.
- Preserve named presets by delegating to existing `WindowSpec`/`make_splitter` paths when no splitter override is provided.
- Add `SearchSpec.method="information_criterion"` with `criterion="aic"|"bic"` as an explicit peer route; keep existing AR/FAR model-owned `selection_method` routing intact.
- IC route uses the existing retune flag/cache path in `_fit_one_model_at_origin`; no validation split required.
- Explicit fold boundaries are end-exclusive integer positions; date labels map to the position just after the label. Example: `[8, 13, 19, 24]` means initial train `0:8` and validation blocks `8:13`, `13:19`, `19:24`.
- `within_fold="expanding"` expands each validation block into per-observation splits with train `0:val_pos` and validation `[val_pos]`.
- IC support boundary remains fitted estimators exposing `ssr_`, `nobs_`, and `n_params_`; generic estimator residual diagnostics are not enough because they do not reliably expose degrees of freedom.

## Gate log

- `~/project/macroforecast/.venv/bin/python -m pytest tests/model_selection/test_validation_splitters.py tests/model_selection/test_information_criterion.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider`
  - First run failed: IC all-failed trial rows lacked `nobs`/`n_params` columns; AIC test DGP overfit to 12 lags.
  - Fixed by adding empty IC failure columns and pinning a deterministic AR(2) DGP where AIC selects lag 2.
  - Rerun passed: 12 passed in 2.95s.
- `~/project/macroforecast/.venv/bin/python -m pytest tests/pipeline/test_result_store.py -k "validation_splitter or digest_tracks" --timeout=300 --timeout-method=thread -q -p no:cacheprovider`
  - Passed: 2 passed, 10 deselected in 0.75s.
- `~/project/macroforecast/.venv/bin/python -m pytest tests/model_selection --timeout=300 --timeout-method=thread -q -p no:cacheprovider`
  - First run failed: generic callable metadata in `SearchSpec.metadata` changed shape because callable JSON handling was too broad.
  - Fixed by limiting callable digest JSON to the `validation_splitter` field.
  - Rerun passed: 52 passed in 4.57s.
- `~/project/macroforecast/.venv/bin/python -m pytest tests/forecasting --timeout=300 --timeout-method=thread -q -p no:cacheprovider`
  - Passed: 160 passed, 65 warnings in 470.16s.
- `~/project/macroforecast/.venv/bin/python -m pytest tests/pipeline --timeout=300 --timeout-method=thread -q -p no:cacheprovider`
  - Passed: 250 passed, 62 warnings in 869.44s.
- `~/project/macroforecast/.venv/bin/python -m mypy macroforecast`
  - First run failed: `_coerce_within_fold` returned `str` instead of the `WithinFoldMode` literal.
  - Fixed by annotating and returning the literal values.
  - Rerun passed: Success, no issues in 112 source files.
- `~/project/macroforecast/.venv/bin/python -m tools.docgen docs/reference`
  - Passed: wrote 37 reference pages.
- `~/project/macroforecast/.venv/bin/python -m tools.docgen --check docs/reference`
  - Passed: docs/reference is up to date.
- `~/project/macroforecast/.venv/bin/python -m pytest tests/reference --timeout=300 --timeout-method=thread -q -p no:cacheprovider`
  - Passed: 20 passed in 1.20s.
- `~/project/macroforecast/.venv/bin/python -m pytest tests/model_selection/test_validation_splitters.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider`
  - Post-mypy-fix rerun passed: 6 passed in 0.77s.
- `~/project/macroforecast/.venv/bin/python -m pytest tests/model_selection/test_information_criterion.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider`
  - Post-review rerun passed: 6 passed in 2.89s.
- `~/project/macroforecast/.venv/bin/python -m mypy macroforecast`
  - Post-review rerun passed: Success, no issues in 112 source files.
- `~/project/macroforecast/.venv/bin/python -m tools.docgen --check docs/reference`
  - Post-review rerun passed: docs/reference is up to date.
- `git diff --check`
  - Passed.
- `git diff -U0 -- macroforecast/pipeline/result_store.py macroforecast/model_selection/search.py macroforecast/model_selection/splitters.py macroforecast/forecasting/policies/base.py | rg -n "^\\+.*except Exception|^\\+.*except:\\s*$"`
  - Passed: no newly added blanket `except`.
