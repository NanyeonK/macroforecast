# Codex Progress - mf-vintagefix

## 2026-07-07 12:42 KST - Initial Read

- Read `.dev-notes/workplan_vintagefix_codex.md` and `.dev-notes/lane_ground_rules.md`.
- Initial status: worktree has untracked lane notes only; no tracked file edits yet.
- Decision: keep all code edits inside the owned vintage runner, `data/vintage.py`, loader/panel validation paths, docs guide, tests, and changelog.
- Decision: implement the `custom_vintages` long-frame fix by documenting the actual accepted grouped-wide snapshot schema and adding validation, not by extending to true long-tidy, because the existing parser and tests already define grouped snapshots and this lane favors bounded onboarding fixes.
- Gate plan: run new/changed test files first, then `tests/forecasting`, then `tests/data`, then `mypy macroforecast`; update this file with every command and result.

## 2026-07-07 12:49 KST - Implementation Decisions

- A1: added a one-time latest-vintage/reference-calendar overlap check in the vintage actual resolver. Empty overlap raises before feature construction; low reference-calendar coverage warns once.
- A2: added `VintagePanelSpec.first_release_max_vintages` (default `12`) and made first-release actual resolution walk forward until a non-missing target observation is found. Missing-forever target dates are counted in `metadata["vintage_source"]["first_release_actuals"]` and warn once per run.
- A3: changed `with_static_extras()` to truncate extras to rows strictly before each `origin_date` before joining. Cache/vintage identity now includes the origin date because the joined extras content can differ for the same base vintage across origins.
- B1: custom mapping vintage keys now fail at `custom_vintages()` construction if timestamp parsing fails; callable-only vintage sources now fail at `VintagePanelSpec(..., actuals_vintage="first_release")` because they do not expose a release calendar.
- B2 deviation: did not refactor `combine()`/`align_frequency()` frequency alignment duplication; only improved the weekly error message and added a recipe smoke test, as instructed.
- B3: duplicate-date panel errors now add a pivot-to-wide hint when a low-cardinality non-numeric column suggests long-format input.
- B4: added `load_custom_csv(na_values=..., date_format=..., dayfirst=...)` with defaults preserving the existing read path when unset.

## 2026-07-07 12:50 KST - Gate Results

- PASS: `~/project/macroforecast/.venv/bin/python -m pytest tests/forecasting/test_vintage_runner.py tests/data/test_vintage.py tests/data/test_combine.py tests/data/test_panel_hardening.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider` -> 52 passed, 8 expected warnings.
- PASS: `~/project/macroforecast/.venv/bin/python -m pytest tests/forecasting --timeout=300 --timeout-method=thread -q -p no:cacheprovider` -> 156 passed, 66 expected warnings.
- PASS: `~/project/macroforecast/.venv/bin/python -m pytest tests/data --timeout=300 --timeout-method=thread -q -p no:cacheprovider` -> 50 passed, 7 expected warnings.
- PASS: `~/project/macroforecast/.venv/bin/python -m mypy macroforecast` -> Success: no issues found in 109 source files.
- PASS: `git diff --check` -> no whitespace errors.
- Docs/CHANGELOG: updated `docs/guide/vintages.md` for FRED month-start calendars, grouped-wide custom snapshots, first-release walks, and static-extra truncation; updated `CHANGELOG.md` under `[Unreleased]`.
