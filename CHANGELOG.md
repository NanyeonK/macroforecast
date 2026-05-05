# Changelog

Notable changes since the v0.0.0 schema reset. See ``CLAUDE.md`` for the
full per-version honesty-pass history embedded in repo documentation.

## [0.5.1] -- 2026-05-05 -- "docs + CI hygiene patch"

### Fixed
* **ci-docs sphinx build**: package docstring in ``macrocast/__init__.py``
  used ``Heading:`` followed by a hyphen-bullet list, which docutils
  parsed as a definition list with unexpected indentation. Switched to
  ``**Heading**`` plus a blank line before each list. Also added
  ``macrocast.scaffold`` to the importable submodule list (it shipped
  in v0.3 but was not advertised in the docstring).
* **ci-docs trigger paths**: docstring-only changes in
  ``macrocast/**`` previously did not re-run ``ci-docs`` (its trigger
  paths were limited to ``docs/**`` + ``pyproject.toml``), so a
  docstring fix could leave a known-broken sphinx build cached on
  ``main``. Added ``macrocast/**`` to the workflow's ``paths`` filter.

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
  subset must execute via ``macrocast.run`` without error. 78 new
  tests.
* **CLI subcommands** (``macrocast run`` / ``replicate`` / ``validate``).
  Previously only ``scaffold`` existed; users had to run
  ``python -c "import macrocast; macrocast.run(...)"`` for everything
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
