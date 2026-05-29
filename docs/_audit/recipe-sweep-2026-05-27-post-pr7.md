# Example Recipe Sweep Audit — 2026-05-27 (post-PR7b)

**macroforecast version**: 0.9.5a0
**Git commit**: a9efe406
**Python version**: 3.12.3
**Run date**: 2026-05-27
**Branch**: `docs-fix/pr7b-recipe-dsl-migration`
**Base**: `docs-fix/pr6-recipe-sweep-audit` @ `d752ecfb`
**Scope**: All `.yaml` files under `examples/recipes/` at depth 1, excluding `goulet_coulombe_2021_replication.yaml`

---

## Summary

| Category | Count |
|----------|-------|
| PASS | 11 |
| FAIL | 0 |
| EXCLUDED (replication) | 1 |
| **Total active** | **11** |

---

## Full Results

| Recipe | Status | cells | manifest |
|--------|--------|-------|----------|
| `examples/recipes/l1_estimated_markov_switching.yaml` | PASS | 1 | True |
| `examples/recipes/l1_minimal.yaml` | PASS | 1 | True |
| `examples/recipes/l1_with_regime.yaml` | PASS | 1 | True |
| `examples/recipes/preprocessing_minimal.yaml` | PASS | 1 | True |
| `examples/recipes/data_preprocessing_minimal.yaml` | PASS | 1 | True |
| `examples/recipes/l4_bagging.yaml` | PASS | 1 | True |
| `examples/recipes/l4_ensemble_ridge_xgb_vs_ar1.yaml` | PASS | 1 | True |
| `examples/recipes/l4_minimal_ridge.yaml` | PASS | 1 | True |
| `examples/recipes/l4_quantile_regression_forest.yaml` | PASS | 1 | True |
| `examples/recipes/l6_full_replication.yaml` | PASS | 1 | True |
| `examples/recipes/l6_standard.yaml` | PASS | 1 | True |

---

## Changes from PR6 audit (2026-05-26)

The five recipes that previously failed with `"L3 uses a step graph (nodes/sinks); fixed_axes sugar is not supported"` (FAIL_SCHEMA Pattern 2) have been migrated to the nodes/sinks DAG pattern in PR7b.

| Recipe | Pre-PR7b status | Post-PR7b status |
|--------|----------------|-----------------|
| `l1_minimal.yaml` | FAIL_SCHEMA | PASS |
| `l1_with_regime.yaml` | FAIL_SCHEMA | PASS |
| `l1_estimated_markov_switching.yaml` | FAIL_SCHEMA | PASS |
| `data_preprocessing_minimal.yaml` | FAIL_SCHEMA | PASS |
| `preprocessing_minimal.yaml` | FAIL_SCHEMA | PASS |

### Migration approach

Each recipe received:
1. A complete `0_meta` block with `failure_policy: fail_fast`, `reproducibility_policy: seeded_reproducible`, and `random_seed: 42`.
2. An updated `1_data` block using `panel_composition: custom_panel_only` with a 12-row `custom_panel_inline` synthetic dataset (dates 2018-01..2018-12, target column named after the recipe's original `target`, one predictor `x1`). Existing L1 axes (`regime_definition`, `regime_estimation_temporal_rule`, etc.) were preserved.
3. A `2_preprocessing` block with safe defaults for custom panel use: `transform_policy: no_transform`, `outlier_policy: none`, `imputation_policy: none_propagate`, `frame_edge_policy: keep_unbalanced`.
4. A new `3_feature_engineering` block using the nodes/sinks DAG: lag-1 features + `target_construction` for h=1.
5. A new `4_forecasting_model` block: ridge (alpha=1.0, direct, expanding, every_origin, min_train_size=6).

### Excluded recipe

`goulet_coulombe_2021_replication.yaml` remains excluded per hard constraint (DO NOT touch). It continues to fail with `unknown L1 axis 'custom_source_policy'` — a one-line axis rename fix deferred to PR8+.

---

## Termination conditions satisfied

- Condition 3: all `examples/recipes/` active recipes (excluding `goulet_coulombe_2021_replication.yaml`) pass `mf.run()` — 11 PASS / 0 FAIL.
- Condition 7: this audit file exists showing PASS 11 / FAIL 0.
