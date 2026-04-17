# Data/Task axes (Phase 3)

Phase 3 introduced 4 new data/task axes (layer `1_data_task`) plus reused/extended 3 prior axes. Together they let recipes declare the methodological choices that govern *what data the forecaster is allowed to see* at every origin date.

| Axis | New / reused | Default | Purpose |
|---|---|---|---|
| `release_lag_rule` | new | `ignore_release_lag` | How publication lag is applied to predictor availability at forecast origin |
| `missing_availability` | new | `complete_case_only` | Policy at data load time when predictors are missing |
| `variable_universe` | new | `all_variables` | Which subset of FRED-MD/QD/SD columns enter X |
| `horizon_list` | new | `arbitrary_grid` | Pre-defined grids of horizons (sweep-friendly) |
| `min_train_size` | reused | `fixed_n_obs` | Minimum training rows required before producing the first forecast |
| `structural_break_segmentation` | reused | `none` | Whether to split the OOS period at known break dates |
| `evaluation_scale` | extended | `original_scale` | Scale on which forecast accuracy is reported |

## YAML usage

```yaml
recipe_id: phase3_demo
data_task:
  release_lag_rule: fixed_lag_all_series
  missing_availability: x_impute_only
  variable_universe: preselected_core
  min_train_size: fixed_n_obs
  structural_break_segmentation: pre_post_covid
  horizon_list: default_1_3_6_12
  evaluation_scale: both
```

## Axis values

### `release_lag_rule`
`ignore_release_lag` (default), `fixed_lag_all_series`, `series_specific_lag`, `calendar_exact_lag`, `lag_conservative`, `lag_aggressive`. See `docs/math/vintage_and_release_lag.md` for the formal mapping.

### `missing_availability`
`complete_case_only` (default), `available_case`, `target_date_drop_if_missing`, `x_impute_only`, `real_time_missing_as_missing`, `state_space_fill`, `factor_fill`, `em_fill`. The first three are operational; the rest are registered placeholders to be wired in v0.6+.

### `variable_universe`
`all_variables` (default), `preselected_core` (INDPRO/PAYEMS/CPIAUCSL/FEDFUNDS/GS10/M2SL/UNRATE), `category_subset`, `paper_replication_subset`, plus 5 advanced subsets (`target_specific_subset`, `expert_curated_subset`, `stability_filtered_subset`, `correlation_screened_subset`, `feature_selection_dynamic_subset`).

### `horizon_list`
`arbitrary_grid` (default — uses `recipe.horizons` directly), `default_1_3_6_12`, `short_only_1_3`, `long_only_12_24`, `paper_specific`. Phase 1 sweep runner accepts `horizon_list` as a sweep axis.

### `min_train_size`
`fixed_n_obs` (default), `fixed_years`, `model_specific_min_train`, `target_specific_min_train`, `horizon_specific_min_train`. Implemented in `macrocast/raw/windowing.py::_resolve_min_train_obs`.

### `structural_break_segmentation`
`none` (default), `pre_post_crisis` (split at 2008-09), `pre_post_covid` (split at 2020-03), `user_break_dates`, `break_test_detected`, `rolling_break_adaptive`. Implemented in `macrocast/raw/windowing.py::compute_train_test_blocks`.

### `evaluation_scale`
`original_scale` (default — successor of legacy `raw_level`), `transformed_scale`, `both`. The legacy `raw_level` is preserved as an alias and silently rewritten to `original_scale` by the compiler.

## Migration from pre-Phase 3 recipes

Recipes that omit any of the 8 axes get the prior runtime behaviour via the defaults above. The only rename is `evaluation_scale: raw_level` → `original_scale`; old YAMLs continue to compile because of the alias entry in `_AXIS_VALUE_ALIASES`.
