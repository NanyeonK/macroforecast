# Example Recipe Sweep Audit — 2026-05-26

**macroforecast version**: 0.9.5a0
**Git commit**: e9538675
**Python version**: 3.12.3
**Run date**: 2026-05-26
**Branch**: `docs-fix/pr6-recipe-sweep-audit`
**Base**: `main` @ `e9538675`
**Scope**: All `.yaml` files under `examples/recipes/` (38 recipes)

---

## Summary

| Category | Count |
|----------|-------|
| PASS | 6 |
| FAIL_SCHEMA | 32 |
| FAIL_RUNTIME | 0 |
| NEGATIVE_EXAMPLE | 0 |
| **Total** | **38** |

---

## Full Results

| Recipe | Status | Error class | Error message (truncated) |
|--------|--------|-------------|--------------------------|
| `goulet_coulombe_2021_replication.yaml` | FAIL_SCHEMA | RuntimeError | cell cell_001 failed: unknown L1 axis 'custom_source_policy' |
| `l0_minimal.yaml` | FAIL_SCHEMA | RuntimeError | cell cell_001 failed: single_target requires leaf_config.target string |
| `l1_5_full.yaml` | FAIL_SCHEMA | RuntimeError | cell cell_001 failed: single_target requires leaf_config.target string |
| `l1_5_minimal.yaml` | FAIL_SCHEMA | RuntimeError | cell cell_001 failed: single_target requires leaf_config.target string |
| `l1_estimated_markov_switching.yaml` | FAIL_SCHEMA | RuntimeError | cell cell_001 failed: L3 uses a step graph (nodes/sinks); fixed_axes sugar is not supported |
| `l1_minimal.yaml` | FAIL_SCHEMA | RuntimeError | cell cell_001 failed: L3 uses a step graph (nodes/sinks); fixed_axes sugar is not supported |
| `l1_with_regime.yaml` | FAIL_SCHEMA | RuntimeError | cell cell_001 failed: L3 uses a step graph (nodes/sinks); fixed_axes sugar is not supported |
| `data_diagnostic_full.yaml` | FAIL_SCHEMA | RuntimeError | cell cell_001 failed: single_target requires leaf_config.target string |
| `data_diagnostic_minimal.yaml` | FAIL_SCHEMA | RuntimeError | cell cell_001 failed: single_target requires leaf_config.target string |
| `preprocessing_minimal.yaml` | FAIL_SCHEMA | RuntimeError | cell cell_001 failed: L3 uses a step graph (nodes/sinks); fixed_axes sugar is not supported |
| `data_preprocessing_minimal.yaml` | FAIL_SCHEMA | RuntimeError | cell cell_001 failed: L3 uses a step graph (nodes/sinks); fixed_axes sugar is not supported |
| `l3_5_full.yaml` | FAIL_SCHEMA | RuntimeError | cell cell_001 failed: single_target requires leaf_config.target string |
| `l3_5_minimal.yaml` | FAIL_SCHEMA | RuntimeError | cell cell_001 failed: single_target requires leaf_config.target string |
| `l3_cascade_pca_on_marx.yaml` | FAIL_SCHEMA | RuntimeError | cell cell_001 failed: single_target requires leaf_config.target string |
| `l3_mccracken_ng_baseline.yaml` | FAIL_SCHEMA | RuntimeError | cell horizon-1 failed: single_target requires leaf_config.target string |
| `l3_minimal_lag_only.yaml` | FAIL_SCHEMA | RuntimeError | cell cell_001 failed: single_target requires leaf_config.target string |
| `l3_multi_pipeline_F_MARX.yaml` | FAIL_SCHEMA | RuntimeError | cell cell_001 failed: single_target requires leaf_config.target string |
| `l4_5_full.yaml` | FAIL_SCHEMA | RuntimeError | cell cell_001 failed: single_target requires leaf_config.target string |
| `l4_5_minimal.yaml` | FAIL_SCHEMA | RuntimeError | cell cell_001 failed: single_target requires leaf_config.target string |
| `l4_bagging.yaml` | PASS | — | cells=1 |
| `l4_ensemble_ridge_xgb_vs_ar1.yaml` | PASS | — | cells=1 |
| `l4_minimal_ridge.yaml` | PASS | — | cells=1 |
| `l4_mrf_placeholder.yaml` | FAIL_SCHEMA | RuntimeError | cell cell_001 failed: single_target requires leaf_config.target string |
| `l4_quantile_regression_forest.yaml` | PASS | — | cells=1 |
| `l4_regime_separate_fit.yaml` | FAIL_SCHEMA | RuntimeError | cell cell_001 failed: unknown L1 axis 'target'; single_target requires leaf_config.target string |
| `l5_full_reporting.yaml` | FAIL_SCHEMA | RuntimeError | cell cell_001 failed: single_target requires leaf_config.target string |
| `l5_latex_export.yaml` | FAIL_SCHEMA | RuntimeError | cell cell_001 failed: single_target requires leaf_config.target string |
| `l5_minimal.yaml` | FAIL_SCHEMA | RuntimeError | cell cell_001 failed: single_target requires leaf_config.target string |
| `l6_full_replication.yaml` | PASS | — | cells=1 |
| `l6_standard.yaml` | PASS | — | cells=1 |
| `l7_coulombe_groups.yaml` | FAIL_SCHEMA | RuntimeError | cell cell_001 failed: single_target requires leaf_config.target string |
| `l7_minimal_shap.yaml` | FAIL_SCHEMA | RuntimeError | cell cell_001 failed: single_target requires leaf_config.target string |
| `l7_multi_method.yaml` | FAIL_SCHEMA | RuntimeError | cell cell_001 failed: single_target requires leaf_config.target string |
| `l7_temporal.yaml` | FAIL_SCHEMA | RuntimeError | cell cell_001 failed: single_target requires leaf_config.target string |
| `l7_transformation_attribution.yaml` | FAIL_SCHEMA | RuntimeError | cell cell_001 failed: single_target requires leaf_config.target string |
| `l8_compact_mode.yaml` | FAIL_SCHEMA | RuntimeError | cell cell_001 failed: single_target requires leaf_config.target string |
| `l8_latex_paper.yaml` | FAIL_SCHEMA | RuntimeError | cell cell_001 failed: single_target requires leaf_config.target string |
| `l8_paper_replication.yaml` | FAIL_SCHEMA | RuntimeError | cell cell_001 failed: single_target requires leaf_config.target string |

---

## Notes

### Classification note: FAIL_SCHEMA vs FAIL_RUNTIME

All 32 failures carry a Python `RuntimeError` wrapper from `execution.py:_run_cells_serial`. However, the inner error messages originate in the schema/validation layer (layer-specific `schema.py` files), not in execution logic. They are classified as **FAIL_SCHEMA** because:

- `"single_target requires leaf_config.target string"` originates in `macroforecast/layers/l1_data/schema.py:633` — a validator gate that fires when the L1 schema requires a string `target` but finds none.
- `"L3 uses a step graph (nodes/sinks); fixed_axes sugar is not supported"` originates in `macroforecast/layers/l3_features/schema.py:81` and `:150` — a structural schema check fired when a recipe supplies `fixed_axes` on a node-graph-based L3 section.
- `"unknown L1 axis 'custom_source_policy'"` originates in `macroforecast/layers/l1_data/schema.py:332` — the validator rejects an axis name that no longer exists in the L1 schema.

The `RuntimeError` wrapping is a presentation artifact of the cell execution loop; the underlying failure occurs during recipe validation before any model fitting begins.

---

### Systematic failures

**Pattern 1 (26 recipes): Missing `1_data` section with `target` in `leaf_config`**

These recipes are partial-layer documentation examples. They demonstrate a single layer's syntax (e.g., `3_feature_engineering:`, `5_evaluation:`, `8_output:`) without providing a full runnable recipe. The runtime needs a `1_data` block with `leaf_config.target` set to a string to resolve the `single_target` constraint. Because the `1_data` section is absent (or present but without `target`), schema validation rejects the recipe.

Affected recipes: `l0_minimal.yaml`, `l1_5_full.yaml`, `l1_5_minimal.yaml`, `data_diagnostic_full.yaml`, `data_diagnostic_minimal.yaml`, `l3_5_full.yaml`, `l3_5_minimal.yaml`, `l3_cascade_pca_on_marx.yaml`, `l3_mccracken_ng_baseline.yaml`, `l3_minimal_lag_only.yaml`, `l3_multi_pipeline_F_MARX.yaml`, `l4_5_full.yaml`, `l4_5_minimal.yaml`, `l4_mrf_placeholder.yaml`, `l4_regime_separate_fit.yaml`, `l5_full_reporting.yaml`, `l5_latex_export.yaml`, `l5_minimal.yaml`, `l7_coulombe_groups.yaml`, `l7_minimal_shap.yaml`, `l7_multi_method.yaml`, `l7_temporal.yaml`, `l7_transformation_attribution.yaml`, `l8_compact_mode.yaml`, `l8_latex_paper.yaml`, `l8_paper_replication.yaml`.

Note: `l4_regime_separate_fit.yaml` also falls into this group. It provides `1_data` with `target: CPIAUCSL` inside `fixed_axes` (an axis-level key) rather than `leaf_config`, and it uses `panel_composition` logic implicitly, triggering both `"unknown L1 axis 'target'"` and `"single_target requires leaf_config.target string"`.

**Pattern 2 (5 recipes): `fixed_axes` used on a node-graph L3 section**

These recipes include a `3_feature_engineering:` block using the `fixed_axes` syntax, but the L3 schema has been migrated to a node/sinks DAG model that rejects `fixed_axes` sugar. The L3 validator fires before execution.

Affected recipes: `l1_estimated_markov_switching.yaml`, `l1_minimal.yaml`, `l1_with_regime.yaml`, `preprocessing_minimal.yaml`, `data_preprocessing_minimal.yaml`.

Note: these recipes show L1 or L2 layer configuration, but they also contain a `3_feature_engineering:` section (likely included for documentation of a multi-layer pattern) whose `fixed_axes` usage is stale.

**Pattern 3 (1 recipe): Renamed L1 axis `custom_source_policy` → `panel_composition`**

`goulet_coulombe_2021_replication.yaml` uses `custom_source_policy` as an L1 axis key in its `fixed_axes` block (`1_data.fixed_axes.custom_source_policy: custom_panel_only`). The current L1 schema uses `panel_composition` as the axis name (confirmed in `macroforecast/layers/l1_data/schema.py:102`). This is a stale axis name from an earlier API version.

---

### Recipes requiring human review

None. All 32 failures follow one of the three patterns above and classify unambiguously as FAIL_SCHEMA. No AMBIGUOUS cases.

---

### Six passing recipes

The six recipes that pass share a structural trait: they are complete, self-contained runnable recipes with all required layers (`0_meta`, `1_data`, `2_preprocessing`, `3_feature_engineering`, `4_forecasting_model`) and a `custom_panel_inline` data block using the current `panel_composition` axis name.

| Recipe | Why it passes |
|--------|---------------|
| `l4_bagging.yaml` | Full recipe, complete layers, current axis names, inline data |
| `l4_ensemble_ridge_xgb_vs_ar1.yaml` | Full recipe, complete layers, current axis names, inline data |
| `l4_minimal_ridge.yaml` | Full recipe, complete layers, current axis names, inline data (reference recipe in CLAUDE.md Quick start) |
| `l4_quantile_regression_forest.yaml` | Full recipe, complete layers, current axis names, inline data |
| `l6_full_replication.yaml` | Full recipe with L5+L6 enabled, current axis names, inline data |
| `l6_standard.yaml` | Full recipe with L5+L6 enabled, current axis names, inline data |

---

### Fix strategy candidates (PR7+ scope)

**FAIL_SCHEMA Pattern 1 — partial-layer recipes missing `1_data` + `target`**

Two fix strategies are possible for PR7+:

Option A (preferred): Convert partial-layer recipes to full runnable recipes. Each recipe that demonstrates a specific layer's syntax should be wrapped in a minimal but runnable recipe skeleton (matching the `l4_minimal_ridge.yaml` pattern: `0_meta`, `1_data` with `custom_panel_inline`, `2_preprocessing`, `3_feature_engineering`, then the layer of interest). This makes every example recipe testable via `mf.run()`.

Option B: Add a `# NOTE: partial-layer snippet — not directly runnable` annotation at the top of each partial recipe and update documentation to distinguish runnable examples from syntax snippets. This avoids recipe edits but leaves 26 of 38 recipes permanently non-runnable.

The 6 currently-passing recipes all follow Option A's pattern. Option A is the higher-fidelity documentation path.

**FAIL_SCHEMA Pattern 2 — `fixed_axes` on node-graph L3**

The 5 affected recipes have a `3_feature_engineering:` section using `fixed_axes` syntax, which was removed when L3 migrated to the node/sinks DAG model. Fix: replace the `fixed_axes`-style L3 block with the standard `nodes`/`sinks` DAG pattern (as shown in the passing recipes). This is a mechanical recipe update with no source changes required.

**FAIL_SCHEMA Pattern 3 — renamed L1 axis `custom_source_policy`**

Fix: rename `custom_source_policy` to `panel_composition` in `goulet_coulombe_2021_replication.yaml`'s `1_data.fixed_axes` block. One-line change.

---

### Next cascade preview

Based on this audit, PR7 remediation shape:

- Estimated PR count: 1 to 3 (depending on whether partial-layer recipes are converted to full runnables or annotated as snippets)
- Scope:
  - 1 axis rename (`custom_source_policy` → `panel_composition`): 1 recipe
  - 5 `fixed_axes` → `nodes/sinks` migrations: 5 recipes (l1_minimal, l1_with_regime, l1_estimated_markov_switching, data_preprocessing_minimal, preprocessing_minimal)
  - 26 partial-layer recipes: either full-recipe conversion (Option A) or snippet annotation (Option B) — this is the major decision gate for PR7 planning
- No source-code changes required for any of these fixes; all repairs are in YAML recipe files under `examples/recipes/`
