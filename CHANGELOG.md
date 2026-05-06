# Changelog

Notable changes since the v0.0.0 schema reset. See ``CLAUDE.md`` for the
full per-version honesty-pass history embedded in repo documentation.

## [0.8.8] -- 2026-05-06 -- "user-friendliness pass (docs only)"

Single docs-only release that bundles five documentation upgrades. No
code changes; same 1035 tests; bit-exact replicate contract unchanged.

### Added
* **`docs/for_recipe_authors/custom_hooks.md` deep dive** -- every one
  of the five extension points (`custom_model`, `custom_preprocessor`,
  `target_transformer`, `custom_feature_block`,
  `custom_feature_combiner`) gets seven sections: decorator usage,
  required signature with type hints, input contract, output contract,
  worked example, common errors, and (for `custom_preprocessor`) a
  table comparing `applied_at='l2'` (pre-pipeline) vs `'l3'`
  (post-pipeline) covering leaf_config keys, runtime stages,
  cleaning_log entries, since-versions.
* **`docs/for_recipe_authors/partial_layer_execution.md` (new)** --
  user guide for running L1 / L2 / L3 / L4 / L5 in isolation via
  `materialize_l1` / `materialize_l2` / `materialize_l3_minimal` /
  `materialize_l4_minimal` / `materialize_l5_minimal` /
  `execute_l1_l2` / `execute_minimal_forecast` / `execute_node`. 9
  runnable snippets + schema tables for nine intermediate artifacts
  (L1DataDefinitionArtifact through L5EvaluationArtifact), with
  debugging use cases (outlier-policy inspection, L3 method-dev
  iteration).
* **`docs/troubleshooting.md` (new)** -- 10 common error scenarios
  with fixes: missing `leaf_config.target`, stale `pip install
  macroforecast` cache, `compare_models().compare()` chain on
  pre-`fit_main` versions, `replicate().sink_hashes_match=False`
  debugging, custom callable not registered,
  `mixed_frequency_representation` gate, missing extras,
  Encyclopedia drift CI failures, partial-layer inspection,
  where-to-ask. Linked from `docs/index.md` "Pick your path".

### Fixed (Simple Docs accuracy)
* **`ExperimentRunResult` / `ExperimentSweepResult` -> `ForecastResult`**
  across 5 simple_api/ pages (the v0.8.0+ rename was incomplete).
* **`result.variants` -> `result.metrics`** and
  **`result.compare("mse")` -> `result.ranking` /
  `result.mean(metric="mse")`** in 4 pages -- aligned with the actual
  v0.8.5 rich-accessor API.

### Fixed (Architecture page drift)
* **`docs/architecture/layer2/index.md`** -- the "Decision order"
  table listed 13 axis names from the pre-0.0-restart 8-layer
  registry (`tcode_policy`, `target_lag_block`,
  `factor_feature_block`, `level_feature_block`,
  `temporal_feature_block`, `rotation_feature_block`,
  `feature_block_combination`, `feature_selection_policy`,
  `feature_selection_semantics`, `feature_builder`,
  `x_lag_feature_block`, `target_normalization`,
  `horizon_target_construction`) that no longer exist in the L2
  `LayerImplementationSpec`. Rewritten as the actual 5 sub-layer ×
  15 axis table (`mixed_frequency_representation`,
  `sd_tcode_policy`, etc.) with an L2 custom-hook section pointing
  at `for_recipe_authors/custom_hooks.md`.
* **Stale numbered parent links** (`Parent: [4. Detail (code): Full]`
  / `Previous: [4.0 Layer 0: ...]` / `Next: [4.2 Layer 2: ...]`) on
  L0 / L1 / L2 / L1.* sub-pages collapsed to plain
  `[Architecture]` / `[Layer N: ...]` (the v0.6.3 number-prefix
  cleanup missed the parent-link strings).

### Notes
* No code, test, or schema changes; encyclopedia tree unchanged
  (drift CI green).
* The agent dispatch contracts for `custom_feature_block` and
  `custom_feature_combiner` documented here are the *actual* runtime
  contracts (`fn(frame, params)` / `fn(inputs, params)`), which
  replace the v0.1-era `FeatureBlockCallableContext` /
  `FeatureCombinerCallableContext` framing that was inaccurate after
  the 0.0 restart.
* `Experiment(...)` constructor today only drives official FRED
  datasets; custom inline panels still need to use
  `mf.run(yaml_recipe)`. This is documented in the worked examples
  and is a follow-up for a future minor.

## [0.8.6] -- 2026-05-06 -- "spec gap fixes: L2 pre-pipeline hook + fit_main + combined-dataset smoke + msfe→mse"

### Added
* **L2 pre-pipeline custom preprocessor hook** (Gap 1) -- new
  ``leaf_config.custom_preprocessor`` slot on L2. Runs *before* the
  canonical transform / outlier / impute / frame_edge stages so users
  can clean the raw L1 panel (drop bad columns, deflation,
  normalisation, custom resampling) before the official t-codes apply.
  Distinct from the v0.2.5 #251 ``custom_postprocessor`` slot which
  runs *after* the canonical pipeline. Both hooks dispatch through the
  same ``macroforecast.custom.register_preprocessor`` contract;
  ``Experiment.use_preprocessor(name, applied_at='l2')`` writes the
  new pre-pipeline slot, ``applied_at='l3'`` (default, unchanged)
  writes the existing post-pipeline slot. ``applied_at`` outside
  ``{'l2', 'l3'}`` raises ``ValueError``. Design Part 2 § L2 carries a
  new "Custom preprocessor hooks (pre vs post pipeline)" subsection
  documenting the two slots side-by-side.
* **Stable ``fit_main`` L4 fit-node id** (Gap 2) --
  ``Experiment.__init__`` and ``Experiment.compare_models([...])`` now
  rename the lone ``fit_<n>_<family>`` node generated by
  ``RecipeBuilder.l4.fit(...)`` to ``fit_main``. Predict-node inputs
  and the L4 ``sinks`` block (``l4_forecasts_v1`` /
  ``l4_model_artifacts_v1``) update atomically. Chained
  ``.compare("4_forecasting_model.nodes.fit_main.params.alpha", [...])``
  follow-ups now have a predictable dotted path independent of the
  original ``model_family=`` argument. Multi-fit (ensemble / horse-race)
  recipes are skipped automatically; an existing ``fit_main`` node
  short-circuits the rename so round-trips through
  ``to_recipe_dict()`` are idempotent.
* **``fred_md+fred_sd`` / ``fred_qd+fred_sd`` combined-dataset smoke**
  (Gap 3) -- new ``tests/integration/test_combined_dataset_smoke.py``
  locks in the L1 dispatch wired in ``_load_official_raw_result`` for
  the two combined-dataset strings. The test pre-populates the raw
  cache with the existing ``tests/fixtures/fred_md_sample.csv`` and
  ``fred_sd_sample.csv`` fixtures so no network access is required;
  the L1 sink is asserted to carry both national columns
  (``INDPRO``, ``RPI``, ``UNRATE``, ``CPIAUCSL``) and regional
  ``VAR_STATE`` columns (``UR_CA``, ``BPPRIVSA_CA``, ``UR_TX``,
  ``BPPRIVSA_TX``). Marked ``slow`` to mirror the existing integration
  suite.

### Changed
* **`primary_metric: msfe` → `primary_metric: mse`** in user-facing
  docs (Gap 4). The L5 schema accepts ``mse`` / ``rmse`` / ``mae``
  etc., not the legacy ``msfe`` alias from the macrocast era. Sweep
  affected ``docs/for_researchers/simple_api/*.md``,
  ``docs/for_recipe_authors/default_profiles.md``, and
  ``docs/navigator/replication_library.md``. The ``inverse_msfe`` /
  ``dmsfe`` L4 combine-method names are unrelated to this rename and
  are kept as-is.
* ``Experiment.use_preprocessor(applied_at='l2')`` no longer raises
  ``NotImplementedError``; the previous "reserved for v0.9" message
  has been removed and the docstring rewritten to document both
  dispatch points side-by-side.
* Simple-API docs (``compare_models.md`` / ``sweep_only_what_you_care.md``
  / ``simple_api/index.md``) reference ``fit_main`` in dotted-path
  examples.

### Tests
* ``tests/api/test_use_methods.py`` -- 3 new tests for the L2
  pre-pipeline hook (leaf_config wiring, end-to-end column-doubler
  preprocessor, ``applied_at`` validation).
* ``tests/api/test_experiment.py`` -- 4 new tests for ``fit_main``
  normalisation (init, ``compare_models``, chained ``.compare`` end-to-end,
  default ``model_family``).
* ``tests/integration/test_combined_dataset_smoke.py`` -- 3 new tests
  for ``fred_md+fred_sd`` / ``fred_qd+fred_sd`` loader dispatch and
  L1-sink merge.

## [0.8.5] -- 2026-05-02 -- "simple API completed (PR 2 of 2): .use_* hooks, ForecastResult rich, variants, two new axes"

### Added
* **`Experiment.use_*` hooks** -- six chainable methods that lower
  user-facing intent into the canonical recipe:
  - ``use_fred_sd_selection(states=, variables=)`` -- writes
    ``state_selection`` / ``sd_variable_selection`` axes (L1.D) plus
    ``selected_states`` / ``selected_sd_variables`` leaf lists.
  - ``use_fred_sd_state_group(group)`` -- L1.D ``fred_sd_state_group``
    axis (16 options, validated).
  - ``use_fred_sd_variable_group(group)`` -- L1.D
    ``fred_sd_variable_group`` axis (12 options, validated).
  - ``use_mixed_frequency_representation(mode)`` -- new L2.A
    ``mixed_frequency_representation`` axis (5 options).
  - ``use_sd_inferred_tcodes()`` -- new L2.B ``sd_tcode_policy=inferred``.
  - ``use_sd_empirical_tcodes(unit, code_map=, audit_uri=)`` -- new
    L2.B ``sd_tcode_policy=empirical`` plus its supporting leaf_config.
  - ``use_preprocessor(name, applied_at='l3')`` -- dispatches a
    ``mf.custom_preprocessor`` registration via the v0.2.5 #251
    runtime hook (``leaf_config.custom_postprocessor``); ``applied_at='l2'``
    raises ``NotImplementedError`` (reserved for v0.9 schema work).
  All return ``self`` for chaining; bad inputs raise ``ValueError``
  with the allowed-options list.
* **`Experiment.variant(name, **overrides)`** -- branches a named
  recipe variant. Variants land under ``recipe['variants'][name]``
  and are expanded to one cell per variant by ``execute_recipe`` in
  ``core/execution.py:_expand_variants``. Variants combine with
  ``compare_models`` / ``compare`` / ``sweep`` axes via the existing
  grid / zip combine modes; cell ids carry a ``__variant-<name>``
  suffix when a variant is active.
* **`ForecastResult` rich accessors** (replaces the v0.8.0 minimal shell):
  - ``forecasts`` -> per-cell ``l4_forecasts_v1`` rows concatenated
    with columns ``cell_id, model_id, target, horizon, origin,
    y_pred, y_pred_lo, y_pred_hi``.
  - ``metrics`` -> per-cell ``l5_evaluation_v1.metrics_table`` with a
    ``cell_id`` column.
  - ``ranking`` -> per-cell ``l5_evaluation_v1.ranking_table`` (empty
    DataFrame when no L5 ranking emitted).
  - ``mean(metric='mse')`` -> per-(model, target, horizon) average of
    one metric; useful one-liner for horse-race summaries.
  - ``read_json(name)`` / ``file_path(name)`` -- per-cell artifact
    accessors, fall back to the manifest root.
  - ``get(cell_id)`` -- pull one cell out by id, raise ``KeyError``
    on miss.
  All return empty DataFrames rather than raising when there is
  nothing to aggregate.
* **`mixed_frequency_representation` axis (L2.A)** -- 5 options:
  ``calendar_aligned_frame`` (default), ``drop_unknown_native_frequency``,
  ``drop_non_target_native_frequency``, ``native_frequency_block_payload``,
  ``mixed_frequency_model_adapter``. Generalises the FRED-SD-specific
  alignment rules to any mixed-frequency panel. Sub-layer L2.A renamed
  from "FRED-SD frequency alignment" to "Mixed frequency alignment".
  Gate: active when dataset includes FRED-SD (or a custom panel
  declares mixed frequency).
* **`sd_tcode_policy` axis (L2.B)** -- 3 options orthogonal to
  ``transform_policy``: ``none`` (default; FRED-SD source values left
  as published), ``inferred`` (national-analog research layer;
  records ``official: false``), ``empirical`` (variable-global /
  state-series stationarity audit map; requires ``sd_tcode_unit``,
  ``sd_tcode_code_map`` when ``unit=state_series``,
  ``sd_tcode_audit_uri``). Gate: active only when dataset includes
  FRED-SD.
* **OptionDoc entries** for each new axis option (8 total: 5 for
  ``mixed_frequency_representation``, 3 for ``sd_tcode_policy``).
* **Encyclopedia regenerated** -- 189 source-tree pages (was 187);
  two new axis pages
  (``docs/encyclopedia/l2/axes/mixed_frequency_representation.md``,
  ``sd_tcode_policy.md``) plus the canonical browse / index updates.
* **Design Part 2** -- ``plans/design/part2_l2_l3_l4.md`` documents
  both new axes (renamed L2.A heading + per-axis sub-section + gate
  table update).
* **Docs**: ``docs/for_researchers/planned_simple_api/`` ->
  ``docs/for_researchers/simple_api/``. Stripped the
  "API status note (current)" planning banner from each page.
  Replaced the index banner with an "every method documented here is
  implemented in v0.8.5" callout. Updated cross-doc links in
  ``docs/for_researchers/index.md`` (toctree + path table).
* **Tests** -- new file ``tests/api/test_use_methods.py`` with
  20 tests covering each ``.use_*`` validator path + variant() alone +
  variant × compare_models × sweep cross-products. Eight rich
  ForecastResult tests added to ``tests/api/test_forecast_result.py``.
  Updated ``test_experiment.py`` variant tests to assert recipe
  emission.

### Changed
* `pyproject.toml` + `macroforecast/__init__.py` -> 0.8.5.
* README + ``docs/install.md`` git pin -> ``@v0.8.5``.

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
