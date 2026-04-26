# Data Handling Policies (1.5)

Declares **how the raw panel is prepared before it reaches the model** — official-frame availability, raw-source missing/outlier treatment before T-codes, release-lag shifts, structural-break handling, and contemporaneous-X rules. Six axes in v1.0 — every value operational.

| Section | axis | Role |
|---|---|---|
| 1.5.1 | [`missing_availability`](#151-missing_availability) | What to do when predictor / target rows contain NaN |
| 1.5.2 | [`raw_missing_policy`](#152-raw_missing_policy) | Whether to repair raw-source missing values before official transforms/T-codes |
| 1.5.3 | [`raw_outlier_policy`](#153-raw_outlier_policy) | Whether to repair raw-source outliers before official transforms/T-codes |
| 1.5.4 | [`release_lag_rule`](#154-release_lag_rule) | How publication lag is modelled when predictors are shifted in time |
| 1.5.5 | [`structural_break_segmentation`](#155-structural_break_segmentation) | Break-dummy augmentation of X around NBER crises / user-supplied break dates |
| 1.5.6 | [`contemporaneous_x_rule`](#156-contemporaneous_x_rule) | Whether X observed at the target date may enter the model |

**Note on dropped axes:**

- `alignment_rule` — mixed-frequency calendar axis; meaningful mainly for FRED-SD. Current runtime uses explicit monthly/quarterly conversion plus provenance reports; a proper MIDAS/state-space mixed-frequency adapter is not first-class yet.
- `evaluation_scale` — re-homed to Layer 2 (`PreprocessContract.evaluation_scale`) where the actual runtime effect lives.
- `exogenous_block` — redundant with `feature_builder` default logic.
- `regime_task` — duplicates 1.3 `oos_period.recession_only_oos` / `expansion_only_oos`.
- `vintage_policy` — dropped as a separate axis. Current real-time vintage control is handled by `information_set_type` plus `leaf_config.data_vintage`, including FRED-SD vintages.
- `x_map_policy` — single-op non-axis; multi-target X mapping is owned by `experiment_unit` (0.2).
**At a glance (defaults):**
- `missing_availability = zero_fill_before_start` — after the selected sample period is sliced, predictor leading missing values before each column's first valid observation are filled with zero and recorded in provenance. Switch to `complete_case_only`, `available_case`, or `x_impute_only` only when a specific missing-data treatment matters.
- `raw_missing_policy = preserve_raw_missing` — leave raw-source missing values unchanged before official transforms/T-codes. Switch only when the research design intentionally cleans raw data before T-code construction.
- `raw_outlier_policy = preserve_raw_outliers` — leave raw-source outliers unchanged before official transforms/T-codes. Switch only when the research design intentionally clips or flags raw data before T-code construction.
- `release_lag_rule = ignore_release_lag` — every column is available at its nominal date. Switch to `fixed_lag_all_series` / `series_specific_lag` when you need to simulate a publication lag.
- `structural_break_segmentation = none` — no break dummies. Switch to `pre_post_crisis` / `pre_post_covid` to add a single NBER-dated break dummy.
- `contemporaneous_x_rule = forbid_contemporaneous` — realistic real-time constraint. Switch to `allow_contemporaneous` only for oracle / data-leak benchmarks.

**Most research runs leave all six at the default.**


---

## 1.5.1 `missing_availability`

**Selects how NaN rows are handled before training.** Four operational values.

### Value catalog

| Value | Status | What it does |
|---|---|---|
| `zero_fill_before_start` | operational | Default. Within the selected sample period, predictor leading missing values are filled with 0. Fully missing predictors are also filled with 0 and warned. Target leading missing dates are reported; target mid-sample missing blocks execution. |
| `complete_case_only` | operational | No panel-level filter; downstream executors handle NaNs per their own policy. |
| `available_case` | operational | Drop rows where any non-date column has NaN before training. Aggressive but legitimate on short fixture windows. |
| `x_impute_only` | operational | Impute predictor (non-target) columns using `leaf_config.x_imputation` ∈ {`mean`, `median`, `ffill`, `bfill`}. Target column retains NaNs so the OOS loop still sees target missingness. |

### Functions & features

- Sample-period availability path: `macrocast.execution.build._apply_sample_period_and_availability(raw_result, recipe, *, target)` implements `zero_fill_before_start` and records `data_reports["availability"]`.
- General missing policy path: `macrocast.execution.build._apply_missing_availability(raw_result, rule, *, target, spec)`.
- Called during dataset loading in `execute_recipe` after official transforms
  have produced the selected frame and before researcher preprocessing runs.
- Compile guard: `x_impute_only` without valid `leaf_config.x_imputation` raises `CompileValidationError`.

### Dropped values

- `target_date_drop_if_missing`, `real_time_missing_as_missing`, `state_space_fill`, `factor_fill`, `em_fill` — complex / niche imputation strategies; v1.1+.

### Recipe usage

```yaml
# Forward-fill predictor columns, keep target NaNs visible
path:
  1_data_task:
    fixed_axes:
      missing_availability: x_impute_only
    leaf_config:
      x_imputation: ffill
```

---

## 1.5.2 `raw_missing_policy`

**Selects raw-source missing treatment before official transforms/T-codes.** Four operational values.

### Value catalog

| Value | Status | What it does |
|---|---|---|
| `preserve_raw_missing` | operational | Default. Leave raw-source missing values untouched before official transforms/T-codes. |
| `zero_fill_leading_x_before_tcode` | operational | Within the selected sample period, fill predictor leading missing values with 0 before official transforms/T-codes. |
| `x_impute_raw` | operational | Impute raw predictor columns before official transforms/T-codes using `leaf_config.raw_x_imputation` in {`mean`, `median`, `ffill`, `bfill`}. |
| `drop_rows_with_raw_missing` | operational | Drop rows with any raw-source missing value before official transforms/T-codes. Aggressive; use only for explicit full-mode designs. |

### Functions & features

- Runtime path: `macrocast.execution.build._apply_raw_missing_policy(raw_result, rule, *, target, spec)`.
- Called before `macrocast.execution.build._apply_tcode_preprocessing(...)`, so any changes affect T-code construction.
- Compile guard: `x_impute_raw` without valid `leaf_config.raw_x_imputation` raises `CompileValidationError`.
- Provenance: runtime records `data_reports["raw_missing"]` with `before_official_transform: true`.

### Recipe usage

```yaml
# Clean raw predictors before official FRED T-codes are applied
path:
  1_data_task:
    fixed_axes:
      raw_missing_policy: x_impute_raw
    leaf_config:
      raw_x_imputation: ffill
```

---

## 1.5.3 `raw_outlier_policy`

**Selects raw-source outlier treatment before official transforms/T-codes.** Six operational values.

### Value catalog

| Value | Status | What it does |
|---|---|---|
| `preserve_raw_outliers` | operational | Default. Leave raw-source outliers untouched before official transforms/T-codes. |
| `winsorize_raw` | operational | Clip raw numeric columns at the 1st and 99th percentiles. |
| `iqr_clip_raw` | operational | Clip raw numeric columns by 1.5 IQR fences. |
| `mad_clip_raw` | operational | Clip raw numeric columns by 3 MAD fences. |
| `zscore_clip_raw` | operational | Clip raw numeric columns by 3 standard deviations. |
| `raw_outlier_to_missing` | operational | Convert values outside the 1st and 99th percentiles to missing before official transforms/T-codes. |

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

**Selects publication-lag policy for predictor shifts.** Three operational values.

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

## 1.5.5 `structural_break_segmentation`

**Selects break-dummy augmentation of the X panel.** Four operational values.

### Value catalog

| Value | Status | What it does |
|---|---|---|
| `none` | operational | Default, no-op. |
| `pre_post_crisis` | operational | Single break dummy at 2008-09-01 (NBER Great-Recession onset). |
| `pre_post_covid` | operational | Single break dummy at 2020-03-01 (NBER COVID-recession onset). |

### Functions & features

- Resolution helper: `macrocast.execution.build._resolve_structural_break_dates(spec)` maps the axis value to a list of break dates.
- The actual augmentation reuses the 1.4 `deterministic_components.break_dummies` path (`augment_array` in `macrocast.execution.deterministic`). Both X_train and X_pred receive the same dummy columns.
- If both `deterministic_components=break_dummies` and `structural_break_segmentation` are set, the augmentations stack (both sets of dummies are added). For user-supplied break dates, use `deterministic_components=break_dummies` with `leaf_config.break_dates` — this is the canonical path after the 2026-04-21 dedup.

### Dropped values

- `break_test_detected`, `rolling_break_adaptive` — change-point detection / adaptive break algorithms; v1.1+.
- `user_break_dates` (2026-04-21) — duplicate of `deterministic_components=break_dummies` + `leaf_config.break_dates`. Both values read the same leaf_config field and dispatched through the same `augment_array(component='break_dummies')` path. Use `deterministic_components=break_dummies` instead.

### Recipe usage

```yaml
# Explicit 2008 + 2020 structural breaks — use deterministic_components
path:
  1_data_task:
    fixed_axes:
      deterministic_components: break_dummies
    leaf_config:
      break_dates: ["2008-09-01", "2020-03-01"]
```

---

## 1.5.6 `contemporaneous_x_rule`

**Selects whether X observed at the target date may enter the model.** Two operational values.

### Value catalog

| Value | Status | What it does |
|---|---|---|
| `forbid_contemporaneous` | operational | Default. `X_pred` is taken at the forecast origin `t` (no contemporaneous observation of y_{t+h}). Realistic real-time forecasting. |
| `allow_contemporaneous` | operational | `X_pred` is taken at the target date `t+h`, aligned with `y_{t+h}`. Oracle / data-leak benchmark used in some comparisons. |

### Functions & features

- Wired inside `macrocast.execution.build._build_raw_panel_training_data` — the axis value selects how `X_train` and `X_pred` align with the target.
- Applies to raw-panel recipes only (autoreg_lagged_target uses target lags, so the axis is irrelevant there).

### Recipe usage

```yaml
# Oracle contemporaneous-X benchmark
path:
  1_data_task:
    fixed_axes:
      contemporaneous_x_rule: allow_contemporaneous
  3_training:
    fixed_axes:
      feature_builder: raw_feature_panel
      model_family: ridge
```

---

## Data Handling Policies (1.5) takeaways

- Every value in every 1.5 axis is operational in v1.0. Zero `registry_only` entries remain.
- All required non-default inputs are compile-time contracts (`release_lag_per_series`, `x_imputation`, `raw_x_imputation`, `break_dates`) and are propagated into `data_task_spec`.
- `raw_missing_policy` and `raw_outlier_policy` run before official transforms/T-codes; `missing_availability` and Layer 2 preprocessing policies run after the official frame exists.
- `structural_break_segmentation` reuses the 1.4 `deterministic_components.break_dummies` augmentation path — the axis just supplies the break-date list.
- `contemporaneous_x_rule` is the only 1.5 axis that affects fit-time X alignment; the other data-handling axes act on the raw or official frame before researcher preprocessing.

Layer 1 per-axis walk complete — 1.1 through 1.5 are all fully honest & operational.
