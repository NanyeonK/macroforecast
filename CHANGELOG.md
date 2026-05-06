# Changelog

Notable changes since the v0.0.0 schema reset. See ``CLAUDE.md`` for the
full per-version honesty-pass history embedded in repo documentation.

## [0.8.0] -- 2026-05-02 -- "core public API: forecast() + Experiment + ForecastResult (PR 1 of 2)"

### Added
* **`mf.forecast(...)`** -- one-shot forecasting helper. Assembles the
  canonical default recipe (L0 fail_fast/seeded_reproducible/serial,
  L1 official-source path with target / horizons / sample window,
  L2 no-transform pass-through, L3 lag1 + target_construction,
  L4 single fit_model node with the requested family, L5 standard mse)
  via ``RecipeBuilder``, runs it through ``execute_recipe``, and wraps
  the result in a :class:`ForecastResult`. Supports ``fred_md``,
  ``fred_qd``, ``fred_sd`` (with explicit ``frequency=``),
  ``fred_md+fred_sd``, ``fred_qd+fred_sd``.
* **`mf.Experiment(...)`** -- builder class for one forecasting study.
  Methods: ``compare_models([f1, f2, ...])``, ``compare(axis_path, values)``,
  ``sweep(axis_path, values)`` (alias of ``compare``),
  ``to_recipe_dict()``, ``to_yaml()``, ``validate()``,
  ``run(output_directory=...)``, ``replicate(manifest_path)``.
  ``compare()`` walks dotted paths into the in-progress recipe dict
  (auto-creates intermediate dicts; addresses L3/L4 ``nodes`` lists by
  the entry's ``id`` field) and replaces the leaf with a
  ``{"sweep": [...]}`` marker.
* **`mf.ForecastResult`** (minimal shell) -- ``cells`` / ``succeeded`` /
  ``manifest_path`` / ``replicate()`` proxies over the underlying
  :class:`ManifestExecutionResult`.
* `tests/api/` -- 32 new tests across ``test_forecast.py``,
  ``test_experiment.py``, ``test_forecast_result.py`` covering: the
  default recipe wiring (L0 / L1 / L4 / L5 axes + start / end +
  horizons), dataset-frequency conflict detection, ``compare_models``
  expansion + ``sweep_values`` recording, generic ``compare()`` /
  ``sweep()`` axis paths, ``to_yaml`` round-trip through
  ``execute_recipe``, ``replicate()`` sink-hash match, ``_set_at``
  edge cases (empty path, traversal into scalar, list-by-id walk,
  sweep-marker overwrite).

### Deferred to v0.8.1 (PR 2 of 2)
* ``Experiment.use_fred_sd_inferred_tcodes()`` /
  ``.use_sd_empirical_tcodes()`` / ``.use_preprocessor()`` /
  ``.use_*`` family of one-call hooks.
* ``Experiment.variant(name, **overrides)`` -- currently raises
  ``NotImplementedError("variant() lands in v0.8.1")``.
* ``ForecastResult`` rich accessors: ``.forecasts`` / ``.metrics`` /
  ``.ranking`` / ``.read_json(...)`` / ``.file_path(...)`` / ``.mean()`` /
  ``.get(...)``.
* Docs migration: drop the ``planned_`` prefix on
  ``docs/for_researchers/planned_simple_api/`` and remove the
  per-page "API status note (current)" banners now that the API
  is real.

### Migration notes
* The new ``mf.forecast()`` / ``mf.Experiment`` surface is **additive** --
  ``mf.run("recipe.yaml")`` / ``mf.replicate(...)`` and the
  ``RecipeBuilder`` continue to work unchanged.
* The exclamation mark in the commit subject (``feat(api)!``) flags the
  new public surface for SemVer awareness; nothing existing is removed
  in v0.8.0.

## [0.7.0] -- 2026-05-06 -- "encyclopedia (replaces auto-emit reference)"

### Added
* **`docs/encyclopedia/`** -- source-committed markdown tree, one page
  per layer / sub-layer / axis (and per-option sections under each axis
  page), with three browse views: by layer, by axis (A-Z), by option
  *value* (A-Z). Generated from the live `LayerImplementationSpec`
  registry plus the `OPTION_DOCS` documentation registry under
  `macroforecast/scaffold/option_docs/`. 187 pages on first emit.
* `macroforecast/scaffold/render_encyclopedia.py` and
  `macroforecast/scaffold/__main__.py` -- the encyclopedia renderer plus
  a `python -m macroforecast.scaffold encyclopedia <out>` entry point.
* `macroforecast scaffold encyclopedia <out>` CLI subcommand on the
  top-level `macroforecast` console script.
* `tests/scaffold/test_render_encyclopedia.py` -- 11 tests covering
  page-count floor, per-layer index + axis pages, browse-by-option
  >= 30 model families, missing-OptionDoc TBD fallback, both CLI
  smoke routes.
* New section in [`docs/encyclopedia/public_api.md`](docs/encyclopedia/public_api.md)
  preserves the curated public Python API table that previously lived
  under `docs/reference/public_api.md`. Linked from
  `for_researchers/index.md` and `for_recipe_authors/index.md`.
* `ci-docs.yml`: new "Encyclopedia drift check" step. Re-emits the
  encyclopedia into a scratch dir and diffs against
  `docs/encyclopedia/`; the build fails if a contributor edits the
  schema or OptionDoc without re-running
  `python -m macroforecast.scaffold encyclopedia docs/encyclopedia/`.
* RELEASE_CHECKLIST.md gains an explicit reminder to regenerate the
  encyclopedia after any OptionDoc / LayerImplementationSpec edit.

### Changed
* `docs/index.md` "Pick your path" row now points at the encyclopedia
  rather than the removed reference index.
* Each `docs/architecture/layer{0..8}/index.md` gained a "See
  encyclopedia" footer cross-link to the matching
  `../../encyclopedia/l{N}/index.md`.
* README.md has a new "Browse the full encyclopedia at
  `docs/encyclopedia/`" pointer in the recipe-gallery section.

### Removed
* `docs/reference/` directory (the previous auto-emitted reference
  tree, including `public_api.md` and the per-build `lN.rst` files
  written by `_emit_optiondoc_reference()` in `docs/conf.py`). The
  curated `public_api.md` content is preserved at
  `docs/encyclopedia/public_api.md`.
* `_emit_optiondoc_reference()` build-time hook in `docs/conf.py`. The
  sphinx build no longer mutates the docs tree -- the encyclopedia is
  now source-committed and CI enforces sync.

### Migration notes
* If you had a local link to `docs/reference/<layer>.rst`, replace it
  with `docs/encyclopedia/<layer_id>/index.md` (per-layer landing) or
  `docs/encyclopedia/<layer_id>/axes/<axis>.md` (per-axis page).
* Bookmarks for `reference/public_api.md` should redirect to
  `encyclopedia/public_api.md`.

## [0.6.3] -- 2026-05-06 -- "openpyxl baseline + FRED-SD docs subdir + architecture number prefix cleanup"

### Changed
* **``openpyxl`` is now a core dependency**, not an optional extra.
  FRED-SD Excel workbook loading is a baseline code path; gating it
  behind ``[excel]`` made the user-visible "first FRED-SD recipe"
  story confusing for negligible install savings (the package itself
  is small). The ``[excel]`` extra is removed from
  ``[project.optional-dependencies]`` and from the ``[all]`` aggregate.

### Fixed (docs)
* **FRED-SD t-code policy pages moved under FRED-SD**:
  ``docs/for_researchers/{fred_sd_transform_policy,
  fred_sd_inferred_tcodes, fred_sd_inferred_tcode_review_v0_1}.md``
  ->
  ``docs/for_researchers/fred_datasets/fred_sd/{transform_policy,
  inferred_tcodes, inferred_tcode_review_v0_1}.md``.
  The flat ``fred_sd.md`` page also moved to
  ``fred_datasets/fred_sd/index.md`` so the SD subtree groups under a
  single page. The toctree at the top of ``fred_datasets/`` keeps
  ``fred_md`` / ``fred_qd`` flat (they have no sub-pages) and points
  at ``fred_sd`` (now a directory).
* **Number prefixes removed from headings**:
  - ``docs/for_researchers/fred_datasets/fred_md.md`` ``# 5.1 FRED-MD``
    -> ``# FRED-MD``; same for QD/SD.
  - ``docs/architecture/layer{0,1,2}/`` headings ``# 4.0 Layer 0`` /
    ``# 4.1 Layer 1`` / ``# 4.1.5 Raw Source Cleaning`` etc. ->
    ``# Layer 0`` / ``# Layer 1`` / ``# Raw Source Cleaning``. The
    folder hierarchy already encodes ordering; the text prefix was
    stale carryover from the pre-reorg flat ``detail/`` tree.
  - Parent links such as ``[5. FRED-Dataset](index.md)`` -> ``[FRED
    datasets](index.md)`` (or ``../index.md`` from ``fred_sd/``).

### Notes
* No code/test changes; same 953 tests, same recipe schema, same
  bit-exact replicate contract. Documentation hygiene release.

## [0.6.2] -- 2026-05-05 -- "docs reorganization (4 audiences + replications track)"

### Changed (docs only)
* **Docs IA reorganized into a 4-audience tree** plus reference and
  replications:

  * ``docs/for_researchers/`` (replaces ``docs/getting_started/`` and
    consolidates ``docs/fred_dataset/``; preserves the planned-API
    preview as ``planned_simple_api/`` inside this tree).
  * ``docs/for_recipe_authors/`` (replaces ``docs/user_guide/`` and
    absorbs ``docs/detail/custom_extensions.md`` ->
    ``custom_hooks.md``, ``target_transformer.md``,
    ``default_profiles.md``).
  * ``docs/for_contributors/`` (replaces ``docs/dev/``).
  * ``docs/architecture/`` (replaces ``docs/detail/`` for the L0-L8
    per-layer pages and the cross-cutting contracts; absorbs the old
    top-level ``foundation_core.md`` -> ``foundation.md`` and
    ``philosophy.md``).
  * ``docs/reference/`` carries the curated ``public_api.md`` (was
    ``docs/api/index.md``); the auto-emitted per-layer
    ``l{0..8}.rst`` pages from
    ``_emit_optiondoc_reference`` already land in this directory.
  * ``docs/replications/`` (NEW in v0.6.1) untouched.
  * ``docs/navigator/`` and ``docs/_html_extra/`` untouched.

* **Top-level ``docs/index.md``** rewritten as a 4-audience picker:
  "Pick your path" maps each role (researcher / recipe author /
  contributor / reference lookup / replications / navigator) to the
  right tree.

### Removed (docs only)
* ``docs/detail/decision_tree_navigator.md`` (redirect-only stub).
* ``docs/detail/contract_source_of_truth.md`` (refers to a registry
  layer that no longer exists).
* ``docs/detail/execution_engine.md`` (refers to the deleted
  ``macroforecast.execution`` legacy stack).
* ``docs/detail/experiment_object.md`` (refers to the deprecated
  ``mc.Experiment`` API that does not ship in v0.6.x).
* ``docs/detail/philosophy.md`` (older duplicate of the top-level
  ``philosophy.md``; kept the more current top-level copy at
  ``architecture/philosophy.md``).

### Archived (docs only)
* ``docs/_archive/post_pr70_runtime_roadmap.md`` (closed phases).
* ``docs/_archive/layer2_revision_plan.md`` (closed phases).
* ``docs/_archive/navigator_ui_redesign_plan.md`` (alternative UI
  track that did not ship).
* ``docs/_archive/source_rst_skeleton_v0.3/`` (orphaned RST skeleton
  pinned to release="0.3.0", unreferenced by ``.readthedocs.yaml``).

  ``docs/_archive/`` is excluded from the Sphinx build via
  ``exclude_patterns``; pages there are kept for blame/history but are
  not part of the live tree.

### Added (CI)
* ``ci-core.yml`` learns a "no stale audience-tree references" check
  that fails the build if any live doc still links to
  ``getting_started/``, ``user_guide/``, ``fred_dataset/``,
  ``simple/``, ``detail/``, ``api/``, or ``dev/`` (the trees the
  reorg removed).

### Notes
* No code changes. Same ``__version__`` lifecycle, same recipe schema,
  same bit-exact replicate contract. v0.6.2 is a documentation-only
  release; the test suite is unchanged at 953 passing.

## [0.6.1] -- 2026-05-05 -- "post-rename consistency sweep"

### Fixed
* **Stale ``v0.5.x`` strings** in ``macroforecast/__init__.py`` docstring,
  ``docs/api/index.md``, ``docs/getting_started/index.md``, and the API
  status banners in every ``docs/simple/*.md`` page. The package
  docstring no longer pins a version number ("**Public surface**" with
  no parenthesised version); API banners use "(current)" /
  "current YAML runtime" instead of "(v0.5.x)".
* **``release.yml`` NOTE comment** still claimed the ``macroforecast``
  PyPI namespace was held by an unrelated 2017 package. Replaced with
  the actual situation: the maintainer owns the namespace and v0.6.0
  was the first release published under it.
* **``.github/RELEASE_CHECKLIST.md`` "PyPI namespace status" section**
  rewritten the same way; gained a ``pip index versions`` post-tag
  check; the throwaway-venv install command now uses
  ``pip install macroforecast==X.Y.Z`` instead of the GitHub-tag fallback.
* **``CLAUDE.md`` header** still showed "Version 0.5.0" + "~785 tests";
  pinned the version to "see ``pyproject.toml`` / ``__version__``" and
  refreshed the test count to 953.

### Notes
* No code changes; same 953 tests, same recipe schema, same bit-exact
  replication contract. v0.6.0 narrative consistency patch.

## [0.6.0] -- 2026-05-05 -- "rename macrocast -> macroforecast"

### Changed (BREAKING)
* **Package rename**: ``macrocast`` -> ``macroforecast``. Both the
  PyPI distribution name and the importable Python module name change.
  ``import macrocast`` no longer resolves; use ``import macroforecast``.
  Convention alias in docs: ``import macroforecast as mf`` (was
  ``as mc``).
* **GitHub repo rename**: ``NanyeonK/macrocast`` ->
  ``NanyeonK/macroforecast``. GitHub auto-redirects old URLs for some
  time but new install commands and bookmarks should use the new name.
* **CLI script**: ``macrocast`` console script renamed to
  ``macroforecast`` (still backed by ``macroforecast.scaffold.cli:main``).
* **PyPI publish unblocked**: the user owns the ``macroforecast``
  PyPI namespace, so ``release.yml`` can now publish successfully on
  tag push (the v0.5.x ``macrocast`` namespace was held by an
  unrelated 2017 package; that warning is gone in v0.6.0).

### Migration

```diff
- import macrocast as mc
+ import macroforecast as mf

- result = mc.run("recipe.yaml")
+ result = mf.run("recipe.yaml")
```

```bash
# Old:
pip install "git+https://github.com/NanyeonK/macrocast.git@v0.5.3"
# New:
pip install macroforecast
# or pinned to a tagged release:
pip install "git+https://github.com/NanyeonK/macroforecast.git@v0.6.0"
```

### Notes
* No runtime behaviour changes; same 953 tests, same recipe schema,
  same bit-exact replication contract. This is a name-only release.
* The ``CHANGELOG`` historical entries below have been swept by sed
  along with the rest of the codebase; references to "macroforecast"
  in v0.5.x entries should be read as "macrocast" historically. The
  past PyPI namespace warnings were about the old ``macrocast`` name
  which is no longer relevant.

## [0.5.3] -- 2026-05-05 -- "version consistency + CI guardrails"

### Fixed
* **README + docs/install.md still pointed at v0.5.1**: ``@v0.5.1`` ->
  ``@v0.5.3`` for every recommended ``pip install
  "git+...@v..."`` command, citation line bumped, version-badge
  removed in favour of real CI badges so the README cannot drift again.
* **``docs/api/index.md`` claimed v0.1**: rewritten to "v0.5.x" and
  framed as a curated reference (it is not actually
  ``sphinx-apidoc`` autogenerated output).
* **``docs/getting_started/index.md`` mis-described Simple Docs**:
  "Existing simple API" / "older high-level Python facade" was
  incorrect — the simple ``mf.forecast`` / ``mf.Experiment`` shape
  is *planned*, not legacy. Row updated to "Planned simple API" and
  redirects to ``macroforecast.run`` / Detail Docs for v0.5.x.
* **API status note only on Simple Docs index**: every other page in
  ``docs/simple/`` (quickstart, run_experiment, compare_models,
  add_custom_model, add_custom_preprocessor, read_results,
  sweep_only_what_you_care, fred_sd) now has the same banner so a
  reader landing via search hits the warning before any sample code.
* **README static badges go stale silently**: replaced the
  ``tests-N passing`` / ``version-X.Y.Z`` static shields with live
  ``ci-core`` and ``ci-docs`` workflow badges. Test count moved to a
  short prose note under the badges.
* **README operational coverage was easy to over-read**: section now
  opens with a callout pointing at
  ``docs/getting_started/runtime_support.md`` for the exact
  end-to-end coverage matrix.
* **``docs/install.md`` core dependencies table missed scipy /
  matplotlib**: added rows so the requirements line and the
  table agree with ``pyproject.toml``.

### Added
* **``ci-core`` version-consistency check**: a step that asserts
  ``pyproject.toml::version`` and ``macroforecast/__init__.py::__version__``
  match, and that no ``@vX.Y.Z`` reference in ``README.md`` /
  ``docs/install.md`` is older than the current package version.
  This is what would have caught v0.5.2 shipping README pointing at
  v0.5.1.
* **``.github/RELEASE_CHECKLIST.md``**: pre-tag / tag / post-tag
  checklist plus the PyPI-namespace status note so future releases
  do not re-discover the same gotchas.
* **``release.yml`` NOTE comment**: explicit reminder that PyPI
  publish is gated on (a) the ``PYPI_API_TOKEN`` secret being
  registered and (b) the token having upload permission for the
  ``macroforecast`` PyPI project (which is currently held by an
  unrelated 2017 package).

### Changed
* This release is functionally equivalent to v0.5.2 — runtime
  behaviour, public API, and test count unchanged. The bump exists
  because v0.5.2 shipped with stale install instructions in its
  README/docs/install.md, and the user-facing fix has to live behind
  a new tag (``bit-exact replication`` ethos: do not silently
  force-push tags).

## [0.5.2] -- 2026-05-05 -- "external review fixes"

### Fixed
* **README install instructions wrong**: ``pip install macroforecast``
  installs an unrelated 2017 package (``macroforecast 0.0.2`` by Amir
  Sani). README + ``docs/install.md`` now warn about the namespace
  collision and recommend ``pip install
  "git+https://github.com/NanyeonK/macroforecast.git@v0.5.2"`` until the
  namespace is resolved.
* **README badges stale**: version ``0.3.0`` -> ``0.5.2``, tests
  ``785 passing`` -> ``953 passing``.
* **docs/install.md placeholders**: ``your-org/macroforecast`` ->
  ``NanyeonK/macroforecast``; expected test count ``291`` -> ``953``;
  ``scipy`` and ``matplotlib`` added to the requirements table to
  match the actual ``pyproject.toml`` dependency set.
* **Simple Docs reference a not-yet-shipped API**: Simple-track pages
  use ``mf.forecast(...)`` and ``mf.Experiment(...).compare_models([
  ...]).run()``; these are not yet exported from
  ``macroforecast.__all__``. Added an "API status note" banner at the top
  of ``docs/simple/index.md`` clarifying that the snippets describe
  the v0.6+ planned wrapper shape, and pointing users to the v0.5.x
  canonical entry surface (``macroforecast.run`` / ``macroforecast.replicate``
  / ``RecipeBuilder`` / ``python -m macroforecast scaffold``).

### Added
* **CLI script entry**: ``[project.scripts]`` now declares
  ``macroforecast = "macroforecast.scaffold.cli:main"`` so installs expose the
  ``macroforecast scaffold`` / ``macroforecast run`` / ``macroforecast replicate``
  / ``macroforecast validate`` shell commands directly. Previously users
  had to use ``python -m macroforecast ...``.

### Changed
* **PyPI publish auth**: ``release.yml`` switched from OIDC Trusted
  Publishing to a ``PYPI_API_TOKEN`` secret since the trusted
  publisher was not registered on PyPI's side. Tag pushes now
  authenticate via the API token in repo secrets.

## [0.5.1] -- 2026-05-05 -- "docs + CI hygiene patch"

### Fixed
* **ci-docs sphinx build**: package docstring in ``macroforecast/__init__.py``
  used ``Heading:`` followed by a hyphen-bullet list, which docutils
  parsed as a definition list with unexpected indentation. Switched to
  ``**Heading**`` plus a blank line before each list. Also added
  ``macroforecast.scaffold`` to the importable submodule list (it shipped
  in v0.3 but was not advertised in the docstring).
* **ci-docs trigger paths**: docstring-only changes in
  ``macroforecast/**`` previously did not re-run ``ci-docs`` (its trigger
  paths were limited to ``docs/**`` + ``pyproject.toml``), so a
  docstring fix could leave a known-broken sphinx build cached on
  ``main``. Added ``macroforecast/**`` to the workflow's ``paths`` filter.

### Notes
* Same code surface and behaviour as v0.5.0; this is a release-hygiene
  bump so the next ``release.yml`` invocation runs against a green
  ``ci-docs`` build.

## [0.5.0] -- 2026-05-05 -- "examples gauntlet + CLI surface"

### Fixed
* **Quick-start broken**: ``examples/recipes/l4_minimal_ridge.yaml``
  (referenced by ``CLAUDE.md`` Quick start) and three other isolated
  layer fragments (``l4_ensemble_ridge_xgb_vs_ar1.yaml`` /
  ``l6_standard.yaml`` / ``l6_full_replication.yaml``) failed with
  ``single_target requires leaf_config.target string``. All four are
  now self-contained end-to-end runnable on a synthetic panel; the
  ensemble + replication recipes use ``gradient_boosting`` (sklearn,
  always installed) instead of ``xgboost`` so they run on a stock
  install -- swap the family in if the corresponding extra is
  installed.
* **L6.B nested test runtime crash**: ``_l6_nested_results`` referenced
  an undefined ``hac_kernel`` local; added the
  ``leaf.get("dependence_correction", "newey_west")`` lookup. Surfaced
  by the ``l6_full_replication`` example which exercises L6.B for the
  first time end-to-end.
* **Wizard wrote the wrong diagnostic-layer recipe keys** under
  ``include_diagnostics=True`` -- ``1_5_data_diagnostics`` etc. instead
  of the runtime's canonical ``1_5_data_summary`` /
  ``2_5_pre_post_preprocessing`` / ``4_5_generator_diagnostics``.
  Fixed in ``scaffold/wizard.py`` ``_LAYER_KEYS``.
* **YAML parse failure**: ``examples/recipes/l1_with_regime.yaml`` was
  a multi-document file. Split into
  ``l1_with_regime.yaml`` + ``l1_estimated_markov_switching.yaml``.

### Added
* **``tests/test_examples_smoke.py``** -- regression guard that
  parametrises over every example yaml. Two layers: every recipe must
  parse + use canonical NEW layer keys; the curated ``runnable``
  subset must execute via ``macroforecast.run`` without error. 78 new
  tests.
* **CLI subcommands** (``macroforecast run`` / ``replicate`` / ``validate``).
  Previously only ``scaffold`` existed; users had to run
  ``python -c "import macroforecast; macroforecast.run(...)"`` for everything
  else.
* **``examples/recipes/goulet_coulombe_2021_replication.yaml``** --
  Goulet-Coulombe (2021) FRED-MD ridge baseline ported to the
  12-layer schema. Runnable end-to-end on the embedded sample panel.

### Changed
* **Archived the v0.0.0 8-layer schema corner**: 13 recipes (9 in
  ``examples/recipes/`` + 3 ``replications/`` + 1 ``templates/``) that
  used the deprecated ``recipe_id`` / ``path: { 1_data_task: ... }``
  wrapper moved to ``examples/recipes/archive_v0/`` with a README
  explaining the migration path. The smoke test skips this directory.
* **``RecipeBuilder`` docstring** now spells out that the per-layer
  namespaces deliberately mutate the shared dict rather than returning
  ``self``; users chain through ``b`` itself, not jQuery-style. No API
  change -- documentation only.

### Test coverage
* 944 passing (up from 866) / 13 skipped.

## [0.3.0] -- 2026-04-XX -- "third honesty pass + new families"

See ``CLAUDE.md`` ``v0.3 third honesty pass`` table.

## [0.25.0] -- 2026-03-XX -- "second honesty pass"

See ``CLAUDE.md`` ``v0.25 second honesty pass`` table.

## [0.2.0] -- 2026-02-XX -- "first honesty pass"

See ``CLAUDE.md`` ``v0.2 honesty-pass demotions`` table.

## [0.1.0] -- 2026-01-XX -- "initial 12-layer release"

Schema-runtime parity for L0..L8 + L1.5/L2.5/L3.5/L4.5 diagnostics.
