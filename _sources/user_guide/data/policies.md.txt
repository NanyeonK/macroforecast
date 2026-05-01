# Forecast-Time And Source-Frame Policies (1.5)

Declares **how forecast-time information and source-frame quality are handled
before Layer 2 sees the data**. Data revision / vintage status is controlled by
`information_set_type` in the forecast-time information page. This page covers
publication lag, same-period x access, raw-source missing/outlier treatment
before T-codes, and frame availability after the Layer 1 source frame exists.

| Section | axis | Role |
|---|---|---|
| 1.5.1 | [`missing_availability`](#151-missing_availability) | Frame Availability Policy: what to do when predictor / target rows contain NaN after the Layer 1 source frame exists |
| 1.5.2 | [`raw_missing_policy`](#152-raw_missing_policy) | Whether to repair raw-source missing values before FRED transforms/T-codes |
| 1.5.3 | [`raw_outlier_policy`](#153-raw_outlier_policy) | Whether to repair raw-source outliers before FRED transforms/T-codes |
| 1.5.4 | [`release_lag_rule`](#154-release_lag_rule) | Publication Lag Rule: when predictor observations are treated as published and usable |
| 1.5.5 | [`contemporaneous_x_rule`](#155-contemporaneous_x_rule) | Same-Period Predictor Rule: whether x observed at the target date may enter the model |

**Note on dropped axes:**

- `alignment_rule` — mixed-frequency calendar axis; meaningful mainly for FRED-SD. Current runtime uses explicit monthly/quarterly conversion plus provenance reports, Layer 2 native-frequency block payloads, custom mixed-frequency adapters, and narrow built-in MIDAS routes (`midas_almon`, `midasr` with `nealmon` / `almonp` / `nbeta` / `genexp` / `harstep`). State-space mixed-frequency likelihoods remain future.
- `evaluation_scale` — re-homed to Layer 2 (`PreprocessContract.evaluation_scale`) where the actual runtime effect lives.
- `exogenous_block` — redundant with `feature_builder` default logic.
- `regime_task` — duplicates 1.3 `oos_period.recession_only_oos` / `expansion_only_oos`.
- `vintage_policy` — dropped as a separate axis. Current data revision / vintage control is handled by `information_set_type` plus `leaf_config.data_vintage`, including FRED-SD vintages.
- `x_map_policy` — single-op non-axis; multi-target X mapping is owned by `study_scope` (0.2).
**At a glance (defaults):**
- `missing_availability = zero_fill_leading_predictor_gaps` — Frame Availability Policy. After the selected sample period is sliced, predictor leading missing values before each column's first valid observation are filled with zero and recorded in provenance. Switch to `require_complete_rows`, `keep_available_rows`, or `impute_predictors_only` only when a specific missing-data treatment matters.
- `raw_missing_policy = preserve_raw_missing` — leave raw-source missing values unchanged before FRED transforms/T-codes. Switch only when the research design intentionally cleans raw data before T-code construction.
- `raw_outlier_policy = preserve_raw_outliers` — leave raw-source outliers unchanged before FRED transforms/T-codes. Switch only when the research design intentionally clips or flags raw data before T-code construction.
- `release_lag_rule = ignore_release_lag` — Publication Lag Rule. Every column is available at its nominal date. Switch to `fixed_lag_all_series` / `series_specific_lag` when you need to simulate a publication lag.
- `contemporaneous_x_rule = forbid_same_period_predictors` — Same-Period Predictor Rule. Realistic real-time constraint. Switch to `allow_same_period_predictors` only for oracle / data-leak benchmarks.

**Most research runs leave all five at the default.**


---

## 1.5.1 `missing_availability`

**Frame Availability Policy.** Selects how NaN rows are handled after the source
frame exists and before Layer 2 representation construction. Four operational
values.

### Value catalog

| Value | Status | What it does |
|---|---|---|
| `zero_fill_leading_predictor_gaps` | operational | Default. Within the selected sample period, predictor leading missing values are filled with 0. Fully missing predictors are also filled with 0 and warned. Target leading missing dates are reported; target mid-sample missing blocks execution. |
| `require_complete_rows` | operational | No panel-level filter; downstream executors handle NaNs per their own policy. |
| `keep_available_rows` | operational | Drop rows where any non-date column has NaN before training. Aggressive but legitimate on short fixture windows. |
| `impute_predictors_only` | operational | Impute predictor (non-target) columns using `leaf_config.x_imputation` ∈ {`mean`, `median`, `ffill`, `bfill`}. Target column retains NaNs so the OOS loop still sees target missingness. |

### Functions & features

- Sample-period availability path: `macrocast.execution.build._apply_sample_period_and_availability(raw_result, recipe, *, target)` implements `zero_fill_leading_predictor_gaps` and records `data_reports["availability"]`.
- General missing policy path: `macrocast.execution.build._apply_missing_availability(raw_result, rule, *, target, spec)`.
- Called during dataset loading in `execute_recipe` after official transforms
  have produced the selected frame and before researcher preprocessing runs.
- Compile guard: `impute_predictors_only` without valid `leaf_config.x_imputation` raises `CompileValidationError`.

### Dropped values

- `target_date_drop_if_missing`, `real_time_missing_as_missing`, `state_space_fill`, `factor_fill`, `em_fill` — complex / niche imputation strategies; v1.1+.

### Recipe usage

```yaml
# Forward-fill predictor columns, keep target NaNs visible
path:
  1_data_task:
    fixed_axes:
      missing_availability: impute_predictors_only
    leaf_config:
      x_imputation: ffill
```

---

## 1.5.2 `raw_missing_policy`

**Selects raw-source missing treatment before FRED transforms/T-codes.** Four operational values.

### Value catalog

| Value | Status | What it does |
|---|---|---|
| `preserve_raw_missing` | operational | Default. Leave raw-source missing values untouched before FRED transforms/T-codes. |
| `zero_fill_leading_predictor_missing_before_tcode` | operational | Within the selected sample period, fill predictor leading missing values with 0 before FRED transforms/T-codes. |
| `impute_raw_predictors` | operational | Impute raw predictor columns before FRED transforms/T-codes using `leaf_config.raw_x_imputation` in {`mean`, `median`, `ffill`, `bfill`}. |
| `drop_raw_missing_rows` | operational | Drop rows with any raw-source missing value before FRED transforms/T-codes. Aggressive; use only for explicit full-mode designs. |

### Functions & features

- Runtime path: `macrocast.execution.build._apply_raw_missing_policy(raw_result, rule, *, target, spec)`.
- Called before `macrocast.execution.build._apply_tcode_preprocessing(...)`, so any changes affect T-code construction.
- Compile guard: `impute_raw_predictors` without valid `leaf_config.raw_x_imputation` raises `CompileValidationError`.
- Provenance: runtime records `data_reports["raw_missing"]` with `before_official_transform: true`.

### Recipe usage

```yaml
# Clean raw predictors before official FRED T-codes are applied
path:
  1_data_task:
    fixed_axes:
      raw_missing_policy: impute_raw_predictors
    leaf_config:
      raw_x_imputation: ffill
```

---

## 1.5.3 `raw_outlier_policy`

**Selects raw-source outlier treatment before FRED transforms/T-codes.** Six operational values.

### Value catalog

| Value | Status | What it does |
|---|---|---|
| `preserve_raw_outliers` | operational | Default. Leave raw-source outliers untouched before FRED transforms/T-codes. |
| `winsorize_raw` | operational | Clip raw numeric columns at the 1st and 99th percentiles. |
| `iqr_clip_raw` | operational | Clip raw numeric columns by 1.5 IQR fences. |
| `mad_clip_raw` | operational | Clip raw numeric columns by 3 MAD fences. |
| `zscore_clip_raw` | operational | Clip raw numeric columns by 3 standard deviations. |
| `set_raw_outliers_to_missing` | operational | Convert values outside the 1st and 99th percentiles to missing before FRED transforms/T-codes. |

### Functions & features

- Runtime path: `macrocast.execution.build._apply_raw_outlier_policy(raw_result, rule, *, spec)`.
- Called before `macrocast.execution.build._apply_tcode_preprocessing(...)`, so any changes affect T-code construction.
- Optional column subset: `leaf_config.raw_outlier_columns`. If omitted, all raw numeric non-date columns are eligible.
- Provenance: runtime records `data_reports["raw_outliers"]` with `before_official_transform: true`.

### Recipe usage

```yaml
# Clip selected raw columns before official FRED T-codes are applied
path:
  1_data_task:
    fixed_axes:
      raw_outlier_policy: iqr_clip_raw
    leaf_config:
      raw_outlier_columns: [INDPRO, RPI]
```

---

## 1.5.4 `release_lag_rule`

**Publication Lag Rule.** Selects publication-lag policy for predictor shifts.
Three operational values.

### Value catalog

| Value | Status | What it does |
|---|---|---|
| `ignore_release_lag` | operational | Default, no-op. Every column is available at its nominal date. |
| `fixed_lag_all_series` | operational | Shift every non-date column by 1 period (uniform 1-month publication lag). |
| `series_specific_lag` | operational | Per-column shift declared via `leaf_config.release_lag_per_series: dict[str, int]`. Columns absent from the dict are left untouched. |

### Functions & features

- Module: `macrocast.execution.build._apply_release_lag(raw_result, rule, *, spec)`.
- Compile guard: `series_specific_lag` without a non-empty `leaf_config.release_lag_per_series` dict raises `CompileValidationError`.

### Dropped values

- `calendar_exact_lag`, `lag_conservative`, `lag_aggressive` — pure duplicates (observable behaviour identical to other values).

### Recipe usage

```yaml
# Typical FRED-MD release convention: most series lag 1 month, UNRATE lags 0
path:
  1_data_task:
    fixed_axes:
      release_lag_rule: series_specific_lag
    leaf_config:
      release_lag_per_series:
        INDPRO: 1
        CPIAUCSL: 1
        UNRATE: 0
```

---

## Moved Out Of Layer 1

`structural_break_segmentation` is now a Layer 2 representation/feature-block decision. It augments the model input with break dummies, so it no longer belongs to the FRED data-frame task. For user-supplied break dates, use Layer 2 `deterministic_components=break_dummies` with `leaf_config.break_dates`.

---

## 1.5.5 `contemporaneous_x_rule`

**Same-Period Predictor Rule.** Selects whether x observed at the target date
may enter the model. Two operational values.

### Value catalog

| Value | Status | What it does |
|---|---|---|
| `forbid_same_period_predictors` | operational | Default. `X_pred` is taken at the forecast origin `t` (no contemporaneous observation of y_{t+h}). Realistic real-time forecasting. |
| `allow_same_period_predictors` | operational | `X_pred` is taken at the target date `t+h`, aligned with `y_{t+h}`. Oracle / data-leak benchmark used in some comparisons. |

### Functions & features

- Wired inside `macrocast.execution.build._build_raw_panel_training_data` — the axis value selects how `X_train` and `X_pred` align with the target.
- Applies to raw-panel recipes only (target_lag_features uses target lags, so the axis is irrelevant there).

### Recipe usage

```yaml
# Oracle contemporaneous-X benchmark
path:
  1_data_task:
    fixed_axes:
      contemporaneous_x_rule: allow_same_period_predictors
  3_feature_engineering:
    nodes:
      - {id: src_x, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: predictors}}}
      - {id: src_y, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: target}}}
      - {id: y_h, type: step, op: target_construction, params: {horizon: 1}, inputs: [src_y]}
    sinks:
      l3_features_v1: {X_final: src_x, y_final: y_h}
```

---

## Forecast-Time And Source-Frame Policies (1.5) takeaways

- Every value in every 1.5 axis is operational in v1.0. Zero `registry_only` entries remain.
- All required non-default inputs are compile-time contracts (`release_lag_per_series`, `x_imputation`, `raw_x_imputation`) and are propagated into `data_task_spec`.
- `raw_missing_policy` and `raw_outlier_policy` run before official transforms/T-codes; `missing_availability` and Layer 2 preprocessing policies run after the Layer 1 source frame exists.
- `contemporaneous_x_rule` is the only 1.5 axis that affects fit-time X alignment; the other data-handling axes act on the raw or source frame before researcher preprocessing.

Layer 1 data-handling walk complete — the current policy axes are operational.
