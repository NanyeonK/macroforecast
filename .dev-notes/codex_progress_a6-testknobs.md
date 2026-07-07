# A6 Test Knobs Progress

Worktree: `/home/nanyeon99/project/mf-a6-testknobs`
Branch: `feat/test-knobs-pairwise`

## Initial context

- Read `.dev-notes/workplan_a6_testknobs_codex.md`.
- Read `.dev-notes/lane_ground_rules.md`.
- No additional design note is named by the workplan.
- Lane progress filename follows the worktree suffix `a6-testknobs`.

## Decisions

- HAC/lag-backed pipeline tests audited from `macroforecast/pipeline/evaluate.py`
  and `macroforecast/tests.py`: `dm`, `cw`, `gw`, `enc_t`, `gr`, and `mz`.
- `mz` already accepts `hac_lags`; it needs spec-build type/range validation.
- `gr` currently exposes `lag_truncate`; the workplan asks for `hac_lags`, so the
  pipeline adapter will accept `hac_lags` as an alias that wins over the horizon
  default while preserving `lag_truncate` for existing callers.
- `cw` is included even though the workplan's example list omits it, because it
  calls the same horizon-derived HAC helper and the acceptance criterion says
  every HAC/lag-based pipeline test.
- Pairwise reporting will be pure post-processing over `report.forecasts` or a
  master forecast frame and will call public functions from `macroforecast.tests`.
- A6-b DM toggle: `dm_test(..., small_sample=True)` preserves the existing
  HLN-corrected statistic and Student-t reference; `small_sample=False` skips
  the HLN factor and uses the plain Diebold-Mariano (1995) asymptotic standard
  normal reference. The legacy `correction="none"` spelling still disables HLN.
- `EvalSpec.test_options["dm"]["small_sample"]` is validated as bool-only at
  spec-build time and threads through the existing pairwise DM option plumbing.

## Gates

- PASS: new/changed tests.
  Command: `~/project/macroforecast/.venv/bin/python -m pytest tests/evaluation/test_tests.py tests/reporting/test_reporting.py tests/pipeline/test_evalspec_threading.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider`
  Result: `91 passed, 1 warning in 5.43s` (warning from existing rescore checkpoint identity path).
- PASS: `tests/evaluation`.
  Command: `~/project/macroforecast/.venv/bin/python -m pytest tests/evaluation --timeout=300 --timeout-method=thread -q -p no:cacheprovider`
  Result: `62 passed in 2.40s`.
- PASS: `tests/reporting`.
  Command: `~/project/macroforecast/.venv/bin/python -m pytest tests/reporting --timeout=300 --timeout-method=thread -q -p no:cacheprovider`
  Result: `25 passed in 3.57s`; rerun after the final pairwise column-normalization patch: `25 passed in 3.46s`.
- PASS: bounded `tests/pipeline`.
  Command: `~/project/macroforecast/.venv/bin/python -m pytest tests/pipeline --timeout=300 --timeout-method=thread -q -p no:cacheprovider`
  Result: `252 passed, 62 warnings in 828.77s (0:13:48)`; warnings are existing ragged-sample, multiprocessing fork, statsmodels frequency, rescore, result-store, and intentional failed-cell warnings.
- PASS: mypy.
  Command: `~/project/macroforecast/.venv/bin/python -m mypy macroforecast`
  Result: `Success: no issues found in 111 source files`; rerun after final source patch stayed clean.
- PASS: docgen regeneration.
  Command: `~/project/macroforecast/.venv/bin/python -m tools.docgen docs/reference`
  Result: `[tools.docgen] wrote 37 pages to docs/reference`.
- PASS: docgen/check.
  Command: `~/project/macroforecast/.venv/bin/python -m tools.docgen --check docs/reference`
  Result: `[tools.docgen] docs/reference is up to date`; rerun after final source patch stayed clean.
- PASS: CHANGELOG update.
  File: `CHANGELOG.md` updated under `[Unreleased]`.

## A6-b DM small-sample toggle gates

- PASS: changed test files.
  Command: `~/project/macroforecast/.venv/bin/python -m pytest tests/evaluation/test_tests.py tests/pipeline/test_evalspec_threading.py tests/parity/test_dm_test.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider`
  Result: `87 passed, 1 warning in 13.04s` (existing rescore checkpoint identity warning).
- PASS: `tests/evaluation`.
  Command: `~/project/macroforecast/.venv/bin/python -m pytest tests/evaluation --timeout=300 --timeout-method=thread -q -p no:cacheprovider`
  Result: `64 passed in 2.27s`.
- PASS: bounded `tests/pipeline`.
  Command: `~/project/macroforecast/.venv/bin/python -m pytest tests/pipeline --timeout=300 --timeout-method=thread -q -p no:cacheprovider`
  Result: `257 passed, 62 warnings in 814.93s (0:13:34)`; warnings match existing ragged-sample, multiprocessing fork, statsmodels frequency, rescore, result-store, and intentional failed-cell warnings.
- PASS: mypy.
  Command: `~/project/macroforecast/.venv/bin/python -m mypy macroforecast`
  Result: `Success: no issues found in 111 source files`.
- PASS: docgen regeneration.
  Command: `~/project/macroforecast/.venv/bin/python -m tools.docgen docs/reference`
  Result: `[tools.docgen] wrote 37 pages to docs/reference`.
- PASS: docgen/check.
  Command: `~/project/macroforecast/.venv/bin/python -m tools.docgen --check docs/reference`
  Result: `[tools.docgen] docs/reference is up to date`.
- PASS: CHANGELOG update.
  File: `CHANGELOG.md` updated under the A6 test knobs `[Unreleased]` block.
