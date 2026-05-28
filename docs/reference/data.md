# Data

`macroforecast.data` has two user-facing jobs:

1. author the recipe's `data` block with `macroforecast.data.data(...)`;
2. load FRED or custom datasets as pandas `DataFrame` objects.

The `data` block decides what panel exists before preprocessing starts. It
selects the source, target, predictor universe, geography, sample window,
horizons, and regime metadata. It does not clean, transform, impute, engineer
features, fit models, or evaluate forecasts.

## Minimal Use

```python
import macroforecast as mf

recipe_part = mf.data.data(
    dataset="fred_md",
    target="CPIAUCSL",
)
```

This returns a recipe fragment:

```python
{
    "data": {
        "fixed_axes": {"dataset": "fred_md"},
        "leaf_config": {"target": "CPIAUCSL"},
    }
}
```

The equivalent YAML is:

```yaml
data:
  fixed_axes:
    dataset: fred_md
  leaf_config:
    target: CPIAUCSL
```

The top-level key is `data`. There is no public `1_data` key in the current
semantic recipe surface.

## Required Choices

Most users choose only the target and, when not using the default FRED-MD
monthly panel, the dataset.

| Use case | Required input |
| --- | --- |
| FRED-MD default | `target` |
| FRED-QD | `dataset="fred_qd"`, `target` |
| FRED-SD only | `dataset="fred_sd"`, `frequency`, `target` |
| custom panel only | `panel_composition="custom_panel_only"`, `frequency`, `target`, and one custom data source |
| official plus custom | `panel_composition="official_plus_custom"`, `custom_source_path`, `custom_merge_rule`, and target settings |
| multiple targets | `targets=[...]`; `target_structure="multi_target"` is inferred |

Custom panel sources can be supplied through `custom_source_path`,
`custom_panel_inline`, or `custom_panel_records`.

## Default Axes

When omitted, these axes resolve as follows:

| Axis | Default |
| --- | --- |
| `panel_composition` | `official_only` |
| `dataset` | `fred_md` |
| `information_set_type` | `final_revised_data` |
| `vintage_policy` | `current_vintage` |
| `target_structure` | `single_target` |
| `variable_universe` | `all_variables` |
| `fred_sd_frequency_policy` | `report_only` |
| `state_selection` | `all_states` |
| `sd_variable_selection` | `all_sd_variables` |
| `missing_availability` | `zero_fill_leading_predictor_gaps` |
| `release_lag_rule` | `ignore_release_lag` |
| `contemporaneous_x_rule` | `allow_same_period_predictors` |
| `target_geography_policy` | `all_states` |
| `predictor_geography_policy` | `match_target` |
| `sample_start_rule` | `max_balanced` |
| `sample_end_rule` | `latest_available` |
| `regime_definition` | `none` |

`frequency` is derived for FRED-MD and FRED-QD families:

| Dataset | Resolved frequency |
| --- | --- |
| `fred_md`, `fred_md+fred_sd` | `monthly` |
| `fred_qd`, `fred_qd+fred_sd` | `quarterly` |
| `fred_sd` | user must set `frequency` |
| custom-only data | user must set `frequency` |

`horizon_set` is also derived from frequency unless explicitly set:

| Frequency | Derived `horizon_set` | Horizons |
| --- | --- | --- |
| `monthly` | `standard_md` | `(1, 3, 6, 12)` |
| `quarterly` | `standard_qd` | `(1, 2, 4, 8)` |

For custom horizons, use `horizon_set="single"` or `"custom_list"` with
`target_horizons`, or `horizon_set="range_up_to_h"` with `max_horizon`.

## Source Axes

| Axis | Choices |
| --- | --- |
| `panel_composition` | `official_only`, `custom_panel_only`, `official_plus_custom` |
| `dataset` | `fred_md`, `fred_qd`, `fred_sd`, `fred_md+fred_sd`, `fred_qd+fred_sd` |
| `frequency` | `monthly`, `quarterly` |
| `information_set_type` | `final_revised_data`, `pseudo_oos_on_revised_data` |
| `vintage_policy` | `current_vintage`, `real_time_alfred` |

`real_time_alfred` currently requires local ALFRED snapshot configuration when
`alfred_mode="local"`.

## Target And Predictor Axes

| Axis | Choices |
| --- | --- |
| `target_structure` | `single_target`, `multi_target` |
| `variable_universe` | `all_variables`, `core_variables`, `category_variables`, `target_specific_variables`, `explicit_variable_list` |
| `missing_availability` | `require_complete_rows`, `keep_available_rows`, `impute_predictors_only`, `zero_fill_leading_predictor_gaps` |
| `release_lag_rule` | `ignore_release_lag`, `fixed_lag_all_series`, `series_specific_lag` |
| `contemporaneous_x_rule` | `allow_same_period_predictors`, `forbid_same_period_predictors` |

Conditional leaf settings:

| Choice | Required leaf setting |
| --- | --- |
| `single_target` | `target` |
| `multi_target` | `targets` |
| `category_variables` | `variable_universe_category_columns`, `variable_universe_category` |
| `target_specific_variables` | `target_specific_columns` |
| `explicit_variable_list` | `variable_universe_columns` |
| `fixed_lag_all_series` | `fixed_lag_periods` |
| `series_specific_lag` | `release_lag_per_series` |

## Geography And FRED-SD

These axes matter when the dataset includes FRED-SD:

| Axis | Choices |
| --- | --- |
| `fred_sd_frequency_policy` | `report_only`, `allow_mixed_frequency`, `reject_mixed_known_frequency`, `require_single_known_frequency` |
| `fred_sd_state_group` | `all_states`, Census regions/divisions, `contiguous_48_plus_dc`, `custom_state_group` |
| `state_selection` | `all_states`, `selected_states` |
| `fred_sd_variable_group` | `all_sd_variables`, domain groups, analog-confidence groups, `custom_sd_variable_group` |
| `sd_variable_selection` | `all_sd_variables`, `selected_sd_variables` |
| `target_geography_policy` | `single_state`, `all_states`, `selected_states` |
| `predictor_geography_policy` | `match_target`, `all_states`, `selected_states`, `national_only` |

Use leaf settings such as `target_state`, `target_states`, `predictor_states`,
`sd_states`, and `sd_variables` when selecting subsets.

## Sample And Regime Axes

| Axis | Choices |
| --- | --- |
| `sample_start_rule` | `earliest_available`, `fixed_date`, `max_balanced` |
| `sample_end_rule` | `latest_available`, `fixed_date` |
| `regime_definition` | `none`, `external_nber`, `external_user_provided`, `estimated_markov_switching`, `estimated_threshold`, `estimated_structural_break` |
| `regime_estimation_temporal_rule` | `expanding_window_per_origin`, `rolling_window_per_origin`, `block_recompute` |

`fixed_date` sample rules require `sample_start_date` or `sample_end_date`.
Partial ISO dates such as `2020` and `2020-03` are accepted and normalized.

Regime settings are metadata at this stage. Estimated regime choices validate
their required inputs and mark runtime estimation work for the later model path.

## Dataset Loaders

Public loaders return pandas frames:

```python
df = mf.data.load_fred_md()
df = mf.data.load_fred_qd()
df = mf.data.load_fred_sd()
df = mf.data.load_custom_csv("panel.csv")
df = mf.data.load_custom_parquet("panel.parquet")
```

Metadata is stored on the frame and accessed through `metadata(...)`:

```python
df = mf.data.load_fred_md()
info = mf.data.metadata(df)
print(info["dataset"])
print(info["artifact"]["cache_hit"])
```

Advanced callers that need the raw envelope can use the `_result` variants:

```python
result = mf.data.load_fred_md_result()
frame = result.data
info = mf.data.metadata(result)
```

The result object keeps the parsed frame, dataset metadata, artifact provenance,
and transform codes together.

## Connection To Other Recipe Blocks

- `meta` sets study-wide execution policy and reproducibility.
- `data` defines the raw panel, target, horizons, and source metadata.
- `preprocessing` receives the materialized data panel and applies cleaning,
  transforms, imputation, and frame-edge rules.

Use `data` for source and target decisions only. Use preprocessing and later
recipe blocks for all transformations after the raw panel exists.
