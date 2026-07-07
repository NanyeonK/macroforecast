# Codex Progress: Issue #421 Docgen Repair

Date: 2026-07-07
Worktree: `/home/nanyeon99/project/mf-docgen`
Branch: `fix/docgen-421`

## Ground Rules Acknowledged

- Work only in this worktree.
- Commit locally only; no push, no `gh`, no PR.
- Stage explicit paths only.
- Use `~/project/macroforecast/.venv/bin/python -m pytest ... --timeout=300 --timeout-method=thread -q -p no:cacheprovider`.
- Keep mypy clean with `~/project/macroforecast/.venv/bin/python -m mypy macroforecast`.
- Update `[Unreleased]` in `CHANGELOG.md`.

## S1 Investigation Map

### Current Failure

- `python -m tools.docgen --help` fails before CLI parsing because `tools/docgen/__init__.py` imports `.introspect`, and `tools/docgen/introspect.py` imports removed modules:
  - `macroforecast.core.layer_specs`
  - `macroforecast.core.ops.registry`
- `.github/workflows/ci-docs.yml` confirms the previous drift gate was removed for the same reason.

### Removed-Architecture Dependencies Under `tools/docgen`

- `introspect.py`: entirely layered-ops registry adapter; delete rather than repair.
- `option_docs/`: hand-written OptionDoc registry for L5-L8 and diagnostics; delete rather than preserve ghosts.
- `render_encyclopedia.py`: renders LayerImplementationSpec/OptionDoc pages, browse-by-layer/axis/option pages, and stale public API text; replace with a current public-API markdown renderer.
- `render_rst.py`: RST emitter for OptionDoc registry; delete.
- `builder.py`, `templates.py`: recipe builder/templates for removed layered YAML runtime; delete from `tools.docgen`.
- `cli.py`, `__main__.py`, `__init__.py`: repair to expose only the current docs/reference generator and `--check` semantics.

### Current Public API Sources

- Top-level lazy surface: `macroforecast/__init__.py`; 22 modules in `_LAZY_MODULES` and hundreds of lazy exports in `_LAZY_EXPORTS`.
- Module surfaces with `__all__`:
  - `macroforecast.meta`
  - `macroforecast.data`
  - `macroforecast.preprocessing`
  - `macroforecast.feature_engineering`
  - `macroforecast.data_analysis`
  - `macroforecast.feature_analysis`
  - `macroforecast.filters`
  - `macroforecast.forecast_analysis`
  - `macroforecast.forecasting`
  - `macroforecast.evaluation`
  - `macroforecast.models`
  - `macroforecast.model_ensemble`
  - `macroforecast.model_selection`
  - `macroforecast.window`
  - `macroforecast.metrics`
  - `macroforecast.tests`
  - `macroforecast.interpretation`
  - `macroforecast.output`
  - `macroforecast.reporting`
  - `macroforecast.pipeline`
- Model registry: `macroforecast.models.list_model_specs()` / `MODEL_SPECS` provide current model metadata, `default_preset`, `input_kind`, search spaces, preprocessing recommendations, and optional extras.

### Reference Pages To Keep Meaningful

- Keep and regenerate current architecture pages under `docs/reference/`: module pages, `custom/*`, index/orientation pages, public API, workflow, and verification/legacy coverage.
- Do not regenerate any L0-L8 option encyclopedia pages or browse-by-layer/axis/option pages. They are absent from the current committed tree and belong to the removed architecture.
- `docs/reference/custom/custom_model.md` must be regenerated or hand-fixed against `macroforecast.models.specs.custom_model`.

### Generator Plan

- Replace `tools.docgen` with a deterministic markdown renderer:
  - introspect current module `__all__` lists;
  - render function-first entries with signature, summary/docstring, parameters, returns, and minimal usage where available;
  - render dataclass/class contracts from dataclass fields and public methods;
  - render `models.md` with current `ModelSpec` registry metadata and current callable signatures;
  - render `public_api.md` from `macroforecast.__all__`;
  - render `custom/*` pages from current custom extension callables;
  - render stable index/orientation pages with current links only.
- Implement `python -m tools.docgen [OUT_DIR]` to write `docs/reference` by default.
- Implement `python -m tools.docgen --check docs/reference` with model-overview-style semantics: render to a temp tree, compare exact contents, print stale/missing/extra paths, and exit non-zero on drift.
- Delete stale files under `tools/docgen` that only served removed layered docs.

## Gate Log

- `git status --short --branch`: on `fix/docgen-421`; untracked `.dev-notes/codex_run.log`, lane files, and this progress file.
- `python -m tools.docgen --help`: FAIL as expected before repair, `ModuleNotFoundError: No module named 'macroforecast.core'`.

## S2 Repair Results

- Replaced `tools.docgen` with a current public-API reference renderer:
  - `python -m tools.docgen` writes the committed `docs/reference` tree.
  - `python -m tools.docgen --check docs/reference` renders and compares exact content.
  - Historical `tools.docgen.render_encyclopedia.write_all` remains as a compatibility import path but now delegates to the current renderer.
- Deleted dead layered-ops internals:
  - `tools/docgen/introspect.py`
  - `tools/docgen/render_rst.py`
  - `tools/docgen/builder.py`
  - `tools/docgen/templates.py`
  - `tools/docgen/option_docs/*`
- Removed all generator imports of removed core ops/layer-spec modules.
- No `docs/reference` page was deleted; all 37 current pages remain meaningful and are regenerated.
- Mypy scope checked in `pyproject.toml`: existing convention is `files = ["macroforecast"]`, so the lane gate uses `python -m mypy macroforecast` and does not extend mypy to `tools/`.

## S3 Material Corrections From Regeneration

Top material corrections observed in the regenerated reference tree:

1. `docs/reference/public_api.md` now lists the live `macroforecast.__all__` instead of stale recipe/core/function namespaces.
2. `docs/reference/public_api.md` now shows all 22 lazy modules, including namespace-only modules such as `feature_diagnostic` and `forecast_diagnostic`.
3. `docs/reference/custom/custom_model.md` now matches `custom_model(...)`: no `metadata=` keyword.
4. `docs/reference/custom/custom_model.md` now documents `default_preset: str = "standard"`.
5. `docs/reference/custom/custom_model.md` now documents `input_kind`.
6. `docs/reference/custom/custom_model.md` now documents `parameters`.
7. `docs/reference/custom/custom_model.md` now documents `requires_scaling`.
8. `docs/reference/custom/custom_model.md` now documents `description`.
9. `docs/reference/models.md` now renders every current `MODEL_SPECS` entry with a `### model_name` section.
10. `docs/reference/models.md` now includes current model registry fields: family, input kind, default preset, default search method, optional extra, scaling requirement, recommended preprocessing, parameters, and search spaces.
11. `docs/reference/data.md` now includes the new `load_fred_md` signature/contract docstring.
12. `docs/reference/data.md` now includes the new `fred_md_vintages` signature/contract docstring.
13. `docs/reference/preprocessing.md` now includes the expanded `preprocess_spec` contract.
14. `docs/reference/feature_engineering.md` now includes the expanded `feature_spec` contract.
15. `docs/reference/pipeline.md` now includes the expanded `TargetSpec` contract.
16. `docs/reference/pipeline.md` now includes the expanded `ResultStore` contract.
17. `docs/reference/legacy_callable_coverage.md` now states that the old layer/option registry pages are removed and replaced by function-first public API docs.
18. `docs/reference/reference_verification.md` now records generated counts from the live package: 21 generated module pages, public symbol totals, callable/class/data counts, and registered model-spec count.
19. Module pages now render deterministic public symbol tables from each module's `__all__`.
20. Generated pages no longer depend on or mention stale generated layer browse pages.

## Part 2 Results

- README rewritten around the current PyPI/package persona:
  - hosted docs link;
  - PyPI and editable install routes;
  - extras list;
  - single-forecast `pipeline_spec` quick use;
  - current capability groups including vintages, result stores, aSPA/uSPA, and LaTeX reporting;
  - link to hosted custom-data tutorial.
- `docs/guide/getting_started.md` now includes PyPI install, extras table, editable install, and a no-checkout smoke snippet.
- Added first-touch docstrings for:
  - `load_fred_md`
  - `feature_spec`
  - `preprocess_spec`
  - `TargetSpec`
  - `ResultStore`
  - `fred_md_vintages`
- Did not touch `load_custom_csv`.
- `docs/conf.py` now has no warning suppressions. Internal scratch plans under `docs/superpowers/plans/*` are excluded from the Sphinx source tree instead of suppressing orphan warnings.
- Fixed exposed link rot: `docs/guide/custom_data_tutorial.md` now links to the generated `paper_accuracy_table` anchor.

## Gate Results

- `~/project/macroforecast/.venv/bin/python -m tools.docgen`: PASS, wrote 37 pages.
- `~/project/macroforecast/.venv/bin/python -m tools.docgen --check docs/reference`: PASS.
- Perturbed-page negative check: PASS via `tests/test_docgen.py::test_docgen_check_fails_on_perturbed_page`.
- `~/project/macroforecast/.venv/bin/python -m pytest tests/test_docgen.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider`: PASS, 3 passed.
- `~/project/macroforecast/.venv/bin/python -m pytest tests/reference tests/data/test_vintage.py tests/data/test_loader_retry.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider`: PASS, 35 passed.
- `~/project/macroforecast/.venv/bin/python -m pytest tests/preprocessing/test_preprocess.py tests/feature_engineering/test_features.py tests/pipeline/test_result_store.py tests/pipeline/test_spec_stage0.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider`: PASS, 138 passed.
- `~/project/macroforecast/.venv/bin/python -m pytest tests/test_docgen.py tests/reference --timeout=300 --timeout-method=thread -q -p no:cacheprovider`: PASS, 23 passed after final generator edits.
- `~/project/macroforecast/.venv/bin/python tools/run_readme_recipe.py`: PASS.
- `~/project/macroforecast/.venv/bin/python tools/gen_model_overview.py --check docs/guide`: PASS, 13 model pages in sync.
- `~/project/macroforecast/.venv/bin/python -m mypy macroforecast`: PASS, no issues in 109 source files.
- Final rerun after changelog/progress edits:
  - `~/project/macroforecast/.venv/bin/python -m pytest tests/test_docgen.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider`: PASS, 5 passed.
  - `.venv-ci-min/bin/python -m pytest tests/test_docgen.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider`: PASS, 5 passed.
  - `~/project/macroforecast/.venv/bin/python -m tools.docgen --check docs/reference`: PASS.
  - `.venv-ci-min/bin/python -m tools.docgen --check docs/reference`: PASS.
  - `.venv-docs-min/bin/python -m tools.docgen --check docs/reference`: PASS.
  - `~/project/macroforecast/.venv/bin/python -m mypy macroforecast`: PASS, no issues in 109 source files.
  - `.venv-ci-min/bin/python -m tools.docgen /tmp/mf-docgen-ci-final2.ze4mvE`: PASS, wrote 37 pages.
  - `~/project/macroforecast/.venv/bin/python -m tools.docgen /tmp/mf-docgen-full-final2.dxJbl1`: PASS, wrote 37 pages.
  - `diff -ru /tmp/mf-docgen-ci-final2.ze4mvE /tmp/mf-docgen-full-final2.dxJbl1`: PASS, no output.
- `~/project/macroforecast/.venv/bin/sphinx-build -W -b html docs docs/_build/html`: PASS.

## PR #456 Env Drift Fix

### Route Chosen

- Route A: make reference generation environment-invariant.
- Diagnosis was written before code edits in `.dev-notes/env_drift_diagnosis.md`.
- Current `HEAD` did not reproduce the brief's `SearchError` diff: `.[ci]`,
  `.[docs]`, and the full venv all generated byte-identical reference trees
  before this fix pass.
- The root-cause class was still present: `tools.docgen` selected module page
  symbols from live runtime `module.__all__`, so a runtime optional-dependency
  export mutation could change generated pages.

### Per-Page Root Causes

- No current page differed in the pre-fix reproduction.
- The observed CI-red class would affect whichever module page saw a runtime
  export-list mutation, plus `reference_verification.md` counts. The example
  `SearchError | macroforecast.model_selection | class` row did not reproduce
  because `macroforecast.model_selection.__all__` currently declares
  `SearchError` unconditionally.

### Fix

- `tools/docgen/renderer.py`: `_public_names()` now reads literal
  source-declared `__all__` names from the module's `.py` file before falling
  back to runtime `__all__`.
- `tests/test_docgen.py`: added a regression test that mutates
  `macroforecast.model_selection.__all__` at runtime and verifies
  `model_selection.md` is unchanged; added a guard that every generated module
  page currently has a source-declared export list.

### Gate Log

- Pre-fix reproduction:
  - `.venv-ci-min/bin/python -m tools.docgen /tmp/mf-docgen-ci-min.rmHATJ`: PASS, wrote 37 pages.
  - `diff -ru docs/reference /tmp/mf-docgen-ci-min.rmHATJ`: PASS, no output.
  - `.venv-docs-min/bin/python -m tools.docgen /tmp/mf-docgen-docs-min.OEAZb3`: PASS, wrote 37 pages.
  - `diff -ru docs/reference /tmp/mf-docgen-docs-min.OEAZb3`: PASS, no output.
  - `~/project/macroforecast/.venv/bin/python -m tools.docgen /tmp/mf-docgen-full.I0Yiat`: PASS, wrote 37 pages.
  - `diff -ru /tmp/mf-docgen-ci-min.rmHATJ /tmp/mf-docgen-full.I0Yiat`: PASS, no output.
  - `diff -ru /tmp/mf-docgen-docs-min.OEAZb3 /tmp/mf-docgen-full.I0Yiat`: PASS, no output.
- `.venv-ci-min/bin/python -m pip install pytest-timeout`: PASS, added only the timeout plugin after reproduction so bounded pytest flags work in the minimal venv.
- `~/project/macroforecast/.venv/bin/python -m pytest tests/test_docgen.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider`: FAIL, 1 failed/4 passed; test fixture needed `raising=False` for a runtime-only monkeypatch.
- `.venv-ci-min/bin/python -m pytest tests/test_docgen.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider`: FAIL, 1 failed/4 passed; same test fixture issue.
- `~/project/macroforecast/.venv/bin/python -m pytest tests/test_docgen.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider`: PASS, 5 passed.
- `.venv-ci-min/bin/python -m pytest tests/test_docgen.py --timeout=300 --timeout-method=thread -q -p no:cacheprovider`: PASS, 5 passed.
- `~/project/macroforecast/.venv/bin/python -m tools.docgen --check docs/reference`: PASS.
- `.venv-ci-min/bin/python -m tools.docgen --check docs/reference`: PASS.
- `.venv-docs-min/bin/python -m tools.docgen --check docs/reference`: PASS.
- Final route-A byte identity:
  - `.venv-ci-min/bin/python -m tools.docgen /tmp/mf-docgen-ci-final.J30UQX`: PASS, wrote 37 pages.
  - `~/project/macroforecast/.venv/bin/python -m tools.docgen /tmp/mf-docgen-full-final.4EvVJW`: PASS, wrote 37 pages.
  - `diff -ru /tmp/mf-docgen-ci-final.J30UQX /tmp/mf-docgen-full-final.4EvVJW`: PASS, no output.
- `~/project/macroforecast/.venv/bin/python -m mypy macroforecast`: PASS, no issues in 109 source files.
