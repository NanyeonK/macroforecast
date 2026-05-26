# Encyclopedia Option Drift Audit — 2026-05-27

**Branch**: `deep-audit/pr5-encyclopedia-drift`
**Method**: Python import of `AxisSpec` options from each `L{N}_LAYER_SPEC`; registry-based extraction of L3/L7 ops; `OPERATIONAL_MODELS | FUTURE_MODELS` for L4; compared against file listing of `docs/reference/encyclopedia/{layer}/{axis}/` option sub-page directories.
**Scope**: Only axes that have per-option sub-page directories are compared (the majority of axes use a single `axes/{axis}.md` page, which is excluded from this drift check). L3 `op/`, L4 `model/`, L6 `equal_predictive_test/`, L6 `nested_test/`, L7 `op/` are all compared.

## Summary

- **Total axes audited**: 13 layers, ~100+ axes total; 12 axes/op-groups have per-option sub-pages
- **Drift-free**: 3 axes/groups (L5 point_metrics, L5 relative_metrics, L5 direction_metrics)
- **CODE_ONLY drift count**: 43 (valid options in code but no dedicated docs sub-page)
- **DOCS_ONLY drift count**: 7 (docs sub-page exists but no corresponding code registration)

## Drift table

| Layer | Axis | Drift type | Option | Classification | Suggested fix | PR scope |
|---|---|---|---|---|---|---|
| l2 | frame_edge_policy | CODE_ONLY | `keep_unbalanced` | DEFERRED | Add option sub-page `l2/frame_edge_policy/keep_unbalanced.md` | follow-up |
| l2 | imputation_policy | CODE_ONLY | `none_propagate` | DEFERRED | Add option sub-page `l2/imputation_policy/none_propagate.md` | follow-up |
| l2 | monthly_to_quarterly_policy | CODE_ONLY | `quarterly_endpoint` | DEFERRED | Add option sub-page | follow-up |
| l2 | monthly_to_quarterly_policy | CODE_ONLY | `quarterly_sum` | DEFERRED | Add option sub-page | follow-up |
| l2 | outlier_policy | CODE_ONLY | `none` | DEFERRED | Add option sub-page `l2/outlier_policy/none.md` | follow-up |
| l2 | quarterly_to_monthly_policy | CODE_ONLY | `chow_lin` | DEFERRED | Add option sub-page `l2/quarterly_to_monthly_policy/chow_lin.md` | follow-up |
| l2 | quarterly_to_monthly_policy | CODE_ONLY | `linear_interpolation` | DEFERRED | Add option sub-page | follow-up |
| l2 | quarterly_to_monthly_policy | CODE_ONLY | `step_forward` | DEFERRED | Add option sub-page | follow-up |
| l2 | transform_policy | CODE_ONLY | `custom_tcode` | DEFERRED | Add option sub-page `l2/transform_policy/custom_tcode.md` | follow-up |
| l2 | transform_policy | CODE_ONLY | `no_transform` | DEFERRED | Add option sub-page `l2/transform_policy/no_transform.md` | follow-up |
| l3 | op | CODE_ONLY | `regime_indicator` | DEFERRED | Add `l3/op/regime_indicator.md` | follow-up |
| l4 | model | CODE_ONLY | `bagging` | DEFERRED | Add `l4/model/bagging.md` | follow-up |
| l4 | model | CODE_ONLY | `decision_tree` | DEFERRED | Add `l4/model/decision_tree.md` | follow-up |
| l4 | model | CODE_ONLY | `macroeconomic_random_forest` | DEFERRED | Add `l4/model/macroeconomic_random_forest.md` | follow-up |
| l4 | model | CODE_ONLY | `quantile_regression_forest` | DEFERRED | Add `l4/model/quantile_regression_forest.md` | follow-up |
| l5 | density_metrics | CODE_ONLY | `crps` | DEFERRED | Add `l5/density_metrics/crps.md` | follow-up |
| l5 | density_metrics | CODE_ONLY | `log_score` | DEFERRED | Add `l5/density_metrics/log_score.md` | follow-up |
| l6 | equal_predictive_test | DOCS_ONLY | `dm_diebold_mariano` | NOTE | AxisSpec `options=()` for L6 is intentional (runtime validated); docs pages are correct | no action |
| l6 | equal_predictive_test | DOCS_ONLY | `dmp_multi_horizon` | NOTE | Same as above | no action |
| l6 | equal_predictive_test | DOCS_ONLY | `gw_giacomini_white` | NOTE | Same as above | no action |
| l6 | equal_predictive_test | DOCS_ONLY | `harvey_newbold_encompassing` | NOTE | Same as above | no action |
| l6 | nested_test | DOCS_ONLY | `clark_west` | NOTE | Same as above | no action |
| l6 | nested_test | DOCS_ONLY | `enc_new` | NOTE | Same as above | no action |
| l6 | nested_test | DOCS_ONLY | `enc_t` | NOTE | Same as above | no action |
| l7 | op | CODE_ONLY | `attention_weights` | DEFERRED | Add `l7/op/attention_weights.md` | follow-up |
| l7 | op | CODE_ONLY | `bootstrap_jackknife` | DEFERRED | Add `l7/op/bootstrap_jackknife.md` | follow-up |
| l7 | op | CODE_ONLY | `bvar_pip` | DEFERRED | Add `l7/op/bvar_pip.md` | follow-up |
| l7 | op | CODE_ONLY | `cumulative_r2_contribution` | DEFERRED | Add `l7/op/cumulative_r2_contribution.md` | follow-up |
| l7 | op | CODE_ONLY | `deep_lift` | DEFERRED | Add `l7/op/deep_lift.md` | follow-up |
| l7 | op | CODE_ONLY | `dual_decomposition` | DEFERRED | Add `l7/op/dual_decomposition.md` | follow-up |
| l7 | op | CODE_ONLY | `fevd` | DEFERRED | Add `l7/op/fevd.md` | follow-up |
| l7 | op | CODE_ONLY | `forecast_decomposition` | DEFERRED | Add `l7/op/forecast_decomposition.md` | follow-up |
| l7 | op | CODE_ONLY | `friedman_h_interaction` | DEFERRED | Add `l7/op/friedman_h_interaction.md` | follow-up |
| l7 | op | CODE_ONLY | `gradient_shap` | DEFERRED | Add `l7/op/gradient_shap.md` | follow-up |
| l7 | op | CODE_ONLY | `group_aggregate` | DEFERRED | Add `l7/op/group_aggregate.md` | follow-up |
| l7 | op | CODE_ONLY | `historical_decomposition` | DEFERRED | Add `l7/op/historical_decomposition.md` | follow-up |
| l7 | op | CODE_ONLY | `integrated_gradients` | DEFERRED | Add `l7/op/integrated_gradients.md` | follow-up |
| l7 | op | CODE_ONLY | `lasso_inclusion_frequency` | DEFERRED | Add `l7/op/lasso_inclusion_frequency.md` | follow-up |
| l7 | op | CODE_ONLY | `lineage_attribution` | DEFERRED | Add `l7/op/lineage_attribution.md` | follow-up |
| l7 | op | CODE_ONLY | `lofo` | DEFERRED | Add `l7/op/lofo.md` | follow-up |
| l7 | op | CODE_ONLY | `mrf_gtvp` | DEFERRED | Add `l7/op/mrf_gtvp.md` | follow-up |
| l7 | op | CODE_ONLY | `orthogonalised_irf` | DEFERRED | Add `l7/op/orthogonalised_irf.md` | follow-up |
| l7 | op | CODE_ONLY | `oshapley_vi` | DEFERRED | Add `l7/op/oshapley_vi.md` | follow-up |
| l7 | op | CODE_ONLY | `pbsv` | DEFERRED | Add `l7/op/pbsv.md` | follow-up |
| l7 | op | CODE_ONLY | `rolling_recompute` | DEFERRED | Add `l7/op/rolling_recompute.md` | follow-up |
| l7 | op | CODE_ONLY | `saliency_map` | DEFERRED | Add `l7/op/saliency_map.md` | follow-up |
| l7 | op | CODE_ONLY | `shap_deep` | DEFERRED | Add `l7/op/shap_deep.md` | follow-up |
| l7 | op | CODE_ONLY | `shap_interaction` | DEFERRED | Add `l7/op/shap_interaction.md` | follow-up |
| l7 | op | CODE_ONLY | `shap_kernel` | DEFERRED | Add `l7/op/shap_kernel.md` | follow-up |
| l7 | op | CODE_ONLY | `transformation_attribution` | DEFERRED | Add `l7/op/transformation_attribution.md` | follow-up |

## Notes on DOCS_ONLY (L6 equal_predictive_test and nested_test)

L6 uses `options=()` in `AxisSpec` for all its axes (including `equal_predictive_test` and `nested_test`). This is by design: L6 validates accepted option values inside `_validate_sub_layer` at runtime, not via the `AxisSpec.options` tuple. The docs sub-pages for these axes (`dm_diebold_mariano`, `dmp_multi_horizon`, `gw_giacomini_white`, `harvey_newbold_encompassing`, `clark_west`, `enc_new`, `enc_t`) are correct and do correspond to real, accepted runtime values. These are NOT true drift — the comparison method cannot detect them as code-side options because AxisSpec does not declare them. This is a known limitation of the audit method.

## Notes on L2 option sub-pages

L2 has per-option sub-page directories for 5 axes: `frame_edge_policy`, `imputation_policy`, `outlier_policy`, `quarterly_to_monthly_policy`, `monthly_to_quarterly_policy`, and `transform_policy`. The CODE_ONLY options for these axes represent options accepted by the code validator (`_validate_options` in `schema.py`) that do not yet have dedicated encyclopedia sub-pages. These were added in v0.25/v0.3 honesty-pass work but the encyclopedia sub-pages were not generated.

## Notes on L3/L4/L7 CODE_ONLY ops/models

- **L3 `regime_indicator`**: registered in `l3_features/ops.py`, no `docs/l3/op/regime_indicator.md`
- **L4 `bagging`, `decision_tree`, `macroeconomic_random_forest`, `quantile_regression_forest`**: listed in `OPERATIONAL_MODELS` in `l4_models/ops.py`, no encyclopedia pages
- **L7 CODE_ONLY (26 ops)**: all registered in `l7_interpretation/schema.py` registry, but only 10 of 36 total L7 ops have encyclopedia pages

## Deprecated axis names check

No references to deprecated axis names found in encyclopedia files:
- `custom_source_policy` (renamed to `panel_composition`): 0 occurrences
- `reproducibility_mode` (renamed to `reproducibility_policy`): 0 occurrences
- `fit_model` (renamed to `fit`): 0 occurrences

## Termination condition

TC8: `test -f docs/_audit/encyclopedia-option-sync-2026-05-27.md` — PASS
