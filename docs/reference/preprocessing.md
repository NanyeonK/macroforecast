# Preprocessing

[Back to reference](index.md)

`macroforecast.preprocessing` defines the recipe block that turns a raw data
panel from [Data](data.md) into the cleaned panel consumed by feature
engineering and models.

The public authoring key is `preprocessing`. It replaces the older numbered key
in hand-written recipes.

## Python Call

```python
import macroforecast as mf

recipe = {
    **mf.data.data(dataset="fred_md", target="INDPRO"),
    **mf.preprocessing.preprocessing(
        transform_policy="apply_official_tcode",
        outlier_policy="mccracken_ng_iqr",
        outlier_action="flag_as_nan",
        imputation_policy="em_factor",
        imputation_temporal_rule="expanding_window_per_origin",
        frame_edge_policy="truncate_to_balanced",
    ),
}
```

`mf.preprocessing.configure(...)` is an alias for
`mf.preprocessing.preprocessing(...)`.

## YAML

```yaml
data:
  fixed_axes:
    dataset: fred_md
  leaf_config:
    target: INDPRO

preprocessing:
  fixed_axes:
    transform_policy: apply_official_tcode
    outlier_policy: mccracken_ng_iqr
    outlier_action: flag_as_nan
    imputation_policy: em_factor
    imputation_temporal_rule: expanding_window_per_origin
    frame_edge_policy: truncate_to_balanced
```

If `preprocessing` is omitted, runtime materialization still applies package
defaults when a full forecast run reaches the cleaning stage.

## Purpose

Preprocessing is the deterministic cleaning stage between raw data and feature
construction. It does five jobs in fixed order:

1. Align mixed-frequency FRED-SD panels when the data source includes FRED-SD.
2. Apply official or custom stationarity transforms.
3. Flag or replace outliers.
4. Impute missing values.
5. Decide how to handle unbalanced frame edges.

It reads `l1_data_definition_v1` internally and produces
`l2_clean_panel_v1`. Those sink names are runtime artifact names; user-facing
recipes should use `data` and `preprocessing` keys.

## Axes

| Axis | Default | Choices |
|------|---------|---------|
| `sd_series_frequency_filter` | `both` | `monthly_only`, `quarterly_only`, `both` |
| `mixed_frequency_representation` | `calendar_aligned_frame` | `calendar_aligned_frame`, `drop_unknown_native_frequency`, `drop_non_target_native_frequency`, `native_frequency_block_payload`, `mixed_frequency_model_adapter` |
| `quarterly_to_monthly_policy` | `step_backward` | `linear_interpolation`, `step_backward`, `step_forward`, `chow_lin` |
| `monthly_to_quarterly_policy` | `quarterly_average` | `quarterly_average`, `quarterly_endpoint`, `quarterly_sum` |
| `transform_policy` | `apply_official_tcode` | `apply_official_tcode`, `no_transform`, `custom_tcode` |
| `sd_tcode_policy` | `none` | `none`, `inferred`, `empirical` |
| `outlier_policy` | `mccracken_ng_iqr` | `mccracken_ng_iqr`, `winsorize`, `zscore_threshold`, `none` |
| `outlier_action` | `flag_as_nan` | `flag_as_nan`, `replace_with_median`, `replace_with_cap_value`, `keep_with_indicator` |
| `imputation_policy` | `em_factor` | `em_factor`, `em_multivariate`, `mean`, `forward_fill`, `linear_interpolation`, `none_propagate` |
| `imputation_temporal_rule` | `expanding_window_per_origin` | `expanding_window_per_origin`, `rolling_window_per_origin`, `block_recompute` |
| `frame_edge_policy` | `truncate_to_balanced` | `truncate_to_balanced`, `drop_unbalanced_series`, `keep_unbalanced`, `zero_fill_leading` |

FRED-SD frequency axes are active only when the data block chooses `fred_sd`,
`fred_md+fred_sd`, or `fred_qd+fred_sd`. If the data source has no FRED-SD
component, those axes are inactive unless explicitly set, in which case
validation rejects them.

## Leaf Config

`leaf_config` carries parameters attached to axis choices.

| Key | Used With | Meaning |
|-----|-----------|---------|
| `custom_tcode_map` | `transform_policy: custom_tcode` | Mapping from series name to t-code integer `1..7`. |
| `sd_tcode_unit` | `sd_tcode_policy: empirical` | Either `variable_global` or `state_series`. |
| `sd_tcode_code_map` | `sd_tcode_policy: empirical`, `sd_tcode_unit: state_series` | Non-empty empirical FRED-SD t-code map. |
| `outlier_iqr_threshold` | `outlier_policy: mccracken_ng_iqr` | Positive IQR multiple; default `10.0`. |
| `winsorize_quantiles` | `outlier_policy: winsorize` | Two increasing probabilities; default `[0.01, 0.99]`. |
| `zscore_threshold_value` | `outlier_policy: zscore_threshold` | Positive z-score cutoff; default `3.0`. |
| `em_n_factors` | `imputation_policy: em_factor` | Positive integer; default `8`. |
| `em_max_iter` | `imputation_policy: em_factor` or `em_multivariate` | Positive integer; default `50`. |
| `em_tolerance` | `imputation_policy: em_factor` or `em_multivariate` | Positive number; default `1e-3`. |
| `rolling_window_size` | `imputation_temporal_rule: rolling_window_per_origin` | Positive integer; default depends on data frequency. |
| `block_recompute_interval` | `imputation_temporal_rule: block_recompute` | Positive integer; default `12` monthly or `4` quarterly. |
| `min_observation_per_series` | `frame_edge_policy` | Non-negative integer. |
| `custom_preprocessor` | runtime hook | Registered pre-cleaning custom preprocessor name. |
| `custom_postprocessor` | runtime hook | Registered post-cleaning custom postprocessor name. |

## Validation Rules

- `custom_tcode` requires `leaf_config.custom_tcode_map`.
- `replace_with_cap_value` requires `outlier_policy: winsorize`.
- `rolling_window_per_origin` with `linear_interpolation` is rejected because
  the current runtime does not implement that combination causally.
- Explicit stateful imputation or outlier policies with `block_recompute`
  produce a soft warning, not a hard error.
- `zero_fill_leading` produces a soft warning because zeros can become a
  learned signal.
- `forward_fill` produces a soft warning because it can create persistence
  artifacts.

## Contract Helpers

The package also exposes `build_preprocess_contract(...)`,
`check_preprocess_governance(...)`, `is_operational_preprocess_contract(...)`,
`preprocess_to_dict(...)`, and `preprocess_summary(...)`.

Those helpers describe preprocessing-governance contracts for experiment
design and custom extensions. They are separate from the runtime
`preprocessing` recipe block above.

## Generated Lookup

For exhaustive per-option prose and standalone operation links, use the
[generated preprocessing lookup](generated/l2/index.md). The generated pages
still use numbered internal names because they are emitted from the live
implementation registry.
