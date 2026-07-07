# Codex Progress: CI Trust Signals + Citation Hygiene

Date: 2026-07-07
Worktree: `/home/nanyeon99/project/mf-citrust`
Branch: `ci/trust-signals`
Lane: `citrust`

## Ground Rules Acknowledged

- Work only in this worktree; do not touch `~/project/macroforecast` except to use its shared venv command path.
- Commit locally only; no push, no `gh`, no PR.
- Stage explicit paths only; never `git add -A` or `git add .`.
- Use bounded pytest invocations with `~/project/macroforecast/.venv/bin/python -m pytest ... --timeout=300 --timeout-method=thread -q -p no:cacheprovider`.
- Run `~/project/macroforecast/.venv/bin/python -m mypy macroforecast`.
- Update `CHANGELOG.md` under `[Unreleased]`.

## Initial State

- `git status --short --branch`: branch `ci/trust-signals`; untracked lane files and `.dev-notes/codex_run.log` already present.
- Initial tag tail recorded before release workflow edits: latest visible tags ended at `v0.9.5a1`.
- Shared venv: Python 3.12.3; optional packages present locally include torch, arch, xgboost, lightgbm, catboost, and shap.

## Decisions

- Use `.dev-notes/codex_progress_citrust.md` as the lane progress file, per worktree suffix.
- Do not edit `README.md`: the workplan explicitly says another lane owns README, and the current README has no badge block to extend.
- Avoid adding a `deep` marker: CI no longer selects with `-m deep`, and removing the stale `not deep` expression from `ci-core` keeps marker algebra aligned with registered markers.
- Keep the seed guard's intent and repoint it to existing package/script paths instead of deleting it.
- Add the docs citation pointer under `docs/guide/citing.md`; note for the README lane: README can later point users to this page and `CITATION.cff`.
- Use `CITATION.cff` release metadata from the current package version `0.9.5` and the matching changelog release date `2026-06-27`; DOI omitted because none exists.
- Add a root `CONTRIBUTING.md` as the lockfile-adjacent development note because no CONTRIBUTING/dev docs file existed.

## Gate Log

- YAML validity: PASS for `.github/workflows/ci-core.yml`, `ci-deep.yml`,
  `ci-extras.yml`, `release.yml`, `.readthedocs.yaml`, and `CITATION.cff`
  using `yaml.safe_load`.
- Seed guard dry check: PASS, no `random_state=42` / `random_seed=42` exact
  literals found under `macroforecast` or `scripts`.
- Nonexistent deep sweep file check: PASS, `tests/core/test_deep_models_sweep_safety.py`
  is absent and no longer referenced.
- Third-party notice path check: PASS, `macroforecast/models/_mrf_reference.py`
  and `macroforecast/models/_mrf_reference.LICENSE` exist; old `_vendor`
  notice path is absent.
- CFF validation: PASS, `uvx cffconvert --validate -i CITATION.cff`.
- Release validation dry-run: PASS, local copy of the workflow script accepts
  `v0.9.5` and matches `pyproject.toml` version `0.9.5`.
- Deep CI selector dry collection: PASS,
  `pytest tests/models tests/forecasting --collect-only ...` collected 334
  tests in 3.31s elapsed with local torch installed.
- Extras CI marker algebra dry collection: PASS,
  `pytest tests/ --collect-only -m 'not slow and not mc and not rparity' ...`
  collected 1201 of 1289 tests, deselecting 88, in 5.19s elapsed. Local timing
  estimate is collection-only because lane rules forbid running the full suite
  in one shot; the weekly workflow stays one job pending real CI runtime.
- Sphinx warning build: PASS,
  `python -m sphinx -W -b html docs docs/_build/citrust-html` succeeded in
  19.86s, so `.readthedocs.yaml` was changed to `fail_on_warning: true`.
- YAML validity rerun after RTD edit: PASS for all changed YAML/CFF files.
- Mypy: PASS, `python -m mypy macroforecast` found no issues in 109 source
  files, elapsed 3.95s.
- Diff whitespace: PASS, `git diff --check`.
- Final tag-list check: PASS, latest visible tags unchanged from the initial
  list and still end at `v0.9.5a1`; no tags were created or pushed.
- Third-party notice parser: PASS, all path-like code spans in
  `THIRD_PARTY_NOTICES.md` resolve.
- Citation version check: PASS, `CITATION.cff` version matches
  `pyproject.toml` version `0.9.5`.
- Stale-reference scan: PASS for live changed files; remaining
  `docs/encyclopedia`, `_vendor`, `OPTION_DOCS`, and `docs/install.md` hits are
  historical changelog entries, not live docs/workflows.
- All planned gates completed before committing.
