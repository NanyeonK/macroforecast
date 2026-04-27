# Benchmark & Predictor Universe (1.4)

Declares **which benchmark to compare against, which predictors the model sees, which raw variables are even in play, and which deterministic features augment the X panel**. Four axes in v1.0 — every value operational via a leaf_config input channel or a simple in-code filter.

| Section | axis | Role |
|---|---|---|
| 1.4.1 | [`benchmark_family`](#141-benchmark_family) | The reference forecast used for relative metrics |
| 1.4.2 | [`predictor_family`](#142-predictor_family) | Which columns of the raw panel are fed to the model |
| 1.4.3 | [`variable_universe`](#143-variable_universe) | Which columns of the raw panel are available in the first place |
| 1.4.4 | [`deterministic_components`](#144-deterministic_components) | Deterministic features appended to X (trend, seasonals, break dummies) |

**Note on dropped values:**

- `predictor_family.text_only` / `mixed_blocks` — require NN/text embeddings stack (v2).
- `variable_universe.feature_selection_dynamic_subset` — CV-in-training feature selection loop; deferred to v1.1 tuning-engine extension.
- `deterministic_components.trend_and_quadratic` — redundant with `linear_trend` + a future `leaf_config.trend_order` channel.

`target_family` (the old 1.4.1 axis) was dropped in PR #32 — subsumed by `target_structure`.
**At a glance (defaults):**
- `benchmark_family` — no default; you always pick one (most studies start with `historical_mean` or `ar_bic`).
- `predictor_family` — feature-builder dynamic default. autoreg_lagged_target → `target_lags_only`; raw_feature_panel → `all_macro_vars`. You rarely set it.
- `variable_universe = all_variables` — the full raw panel is available. Switch to a subset only when the recipe explicitly narrows the candidate variables.
- `deterministic_components = none` — no X augmentation. Switch to `linear_trend` / seasonals / `break_dummies` when your target needs them.

**Most research runs pick `benchmark_family` and leave the other three at the default.**


---

## 1.4.1 `benchmark_family`

**Selects the reference forecast for relative metrics.** All 12 kept values are operational in v1.0. Values that require user-supplied inputs are validated at compile time.

### Value catalog

| Value | Status | What it does |
|---|---|---|
| `historical_mean` | operational | Training-set mean. Default. |
| `zero_change` | operational | Random-walk at `y_t`. |
| `ar_bic` | operational | AR model with BIC-selected lag order. |
| `ar_fixed_p` | operational | AR model at a fixed lag `p` (`benchmark_config.benchmark_fixed_p`). |
| `ardi` | operational | AR + Diffusion Index (factor) model. |
| `rolling_mean` | operational | Rolling-window mean (`benchmark_config.benchmark_window_len`). |
| `custom_benchmark` | operational | Arbitrary callable supplied in `benchmark_config.benchmark_callable`. |
| `expert_benchmark` | operational | Callable supplied in `benchmark_config.expert_callable`. |
| `factor_model` | operational | Single-factor OLS on the leading principal factor (v1.0 self-contained impl). |
| `multi_benchmark_suite` | operational | Runs each member in `leaf_config.benchmark_suite: list[str]` and returns the arithmetic mean. |
| `paper_specific_benchmark` | operational | Pre-computed forecast series supplied via `leaf_config.paper_forecast_series: dict[target → Series]`. |
| `survey_forecast` | operational | Same pattern, `leaf_config.survey_forecast_series`. |

### Functions & features

- `macrocast.execution.build._run_benchmark_executor` dispatches by `benchmark_family` value.
- `factor_model`: z-scored leading-factor regression; falls back to `historical_mean` for training windows < 6 rows.
- `multi_benchmark_suite`: inline dispatch over `leaf_config.benchmark_suite` members (allowed set: historical_mean, zero_change, ar_bic, rolling_mean, ar_fixed_p, ardi). Missing or unsupported members raise `CompileValidationError`.
- `paper_specific_benchmark` / `survey_forecast`: look up the forecast at `train.index[-1] + horizon` months (monthly freq); fall back to the most recent trailing value on miss. The required target-keyed series dict is checked at compile time.
- `expert_benchmark`: programmatic only; requires `leaf_config.benchmark_config.expert_callable`.

### Recipe usage

```yaml
# Paper-replication: compare against the paper's published forecast
path:
  1_data_task:
    leaf_config:
      paper_forecast_series:
        INDPRO: ...   # pd.Series keyed by date
  3_training:
    fixed_axes:
      benchmark_family: paper_specific_benchmark
```

---

## 1.4.2 `predictor_family`

**Selects which columns of the raw panel become model predictors.** 6 operational values.

### Value catalog

| Value | Status | What it does |
|---|---|---|
| `target_lags_only` | operational | Only the target's own lags (forces `feature_builder=autoreg_lagged_target`). Default for autoreg recipes. |
| `all_macro_vars` | operational | Every column except the target. Default for raw-panel recipes. |
| `category_based` | operational | User-supplied category mapping: `leaf_config.predictor_category_columns: dict[str, list[str]]` + `leaf_config.predictor_category`. |
| `factor_only` | operational | Columns whose name starts with `F_` (factor outputs). |
| `explicit_variable_list` | operational | User-supplied column list: `leaf_config.handpicked_columns: list[str]`. |

### Functions & features

- `macrocast.execution.build._raw_panel_columns(frame, target, predictor_family, spec)` dispatches on the rule.
- Target column is always excluded from the predictor set.
- Compile guards: `explicit_variable_list` requires `leaf_config.handpicked_columns`; `category_based` requires `leaf_config.predictor_category_columns` and `leaf_config.predictor_category`.

### Dropped values

- `text_only`: requires text-embedding / NN domain stack — deferred to v2 (Transformer scope).
- `mixed_blocks`: multi-block NN architecture — deferred to v2.

### Recipe usage

```yaml
path:
  1_data_task:
    fixed_axes:
      predictor_family: explicit_variable_list
    leaf_config:
      handpicked_columns: [RPI, UNRATE, CPIAUCSL]
  3_training:
    fixed_axes:
      feature_builder: raw_feature_panel
      model_family: ridge
```

---

## 1.4.3 `variable_universe`

**Selects which columns of the raw panel survive dataset filtering before any training begins.** 8 operational values.

### Value catalog

| Value | Status | What it does |
|---|---|---|
| `all_variables` | operational | Default. No filter. |
| `core_variables` | operational | FRED-MD core macro variables (`_PRESELECTED_CORE` set). |
| `explicit_variable_list` | operational | User-supplied column list: `leaf_config.variable_universe_columns: list[str]`. Consolidates the former paper_replication / expert_curated / stability_filtered / correlation_screened subsets — all four had identical runtime semantics (drop_duplicate cleanup, 2026-04-21). |
| `category_variables` | operational | `leaf_config.variable_universe_category_columns: dict[str, list[str]]` + `leaf_config.variable_universe_category`. |
| `target_specific_variables` | operational | `leaf_config.target_specific_columns: dict[target, list[str]]`. |

### Functions & features

- `macrocast.execution.build._apply_variable_universe(raw_result, rule, spec, target)` is called during dataset loading in `execute_recipe`.
- Target and date columns are always preserved after filtering.
- Runtime discovery (stability / correlation) is out of scope — users supply the subset.
- Compile guards: `explicit_variable_list` requires `leaf_config.variable_universe_columns`; `category_variables` requires `leaf_config.variable_universe_category_columns` and `leaf_config.variable_universe_category`; `target_specific_variables` requires `leaf_config.target_specific_columns` entries for the current target(s).

### Dropped values

- `feature_selection_dynamic_subset`: CV-in-training feature selection loop requires a tuning-engine extension — deferred to v1.1.
- `paper_replication_subset`, `expert_curated_subset`, `stability_filtered_subset`, `correlation_screened_subset` (2026-04-21): four labels shared identical runtime semantics (single `list[str]` input + column filter). Consolidated into `explicit_variable_list`.

### Recipe usage

```yaml
# target-specific subset
path:
  1_data_task:
    fixed_axes:
      variable_universe: target_specific_variables
    leaf_config:
      target_specific_columns:
        INDPRO: [RPI, UNRATE, CPIAUCSL]
        PAYEMS: [UNRATE, AWHMAN, CPIAUCSL]
```

```yaml
# hand-picked column list
path:
  1_data_task:
    fixed_axes:
      variable_universe: explicit_variable_list
    leaf_config:
      variable_universe_columns: [RPI, UNRATE, CPIAUCSL]
```

---

## 1.4.4 `deterministic_components`

**Appends deterministic feature columns to the X matrix.** 6 operational values.

### Value catalog

| Value | Status | What it does |
|---|---|---|
| `none` | operational | Default. No augmentation. |
| `constant_only` | operational | Explicit column of 1s (redundant with `fit_intercept=True` default, but records the intent). |
| `linear_trend` | operational | Adds a `_dc_trend` column (0, 1, …, n-1). |
| `monthly_seasonal` | operational | Adds 11 monthly dummies (`_dc_month_01` … `_dc_month_11`; December is the reference). |
| `quarterly_seasonal` | operational | Adds 3 quarterly dummies (`_dc_q1` … `_dc_q3`; Q4 reference). |
| `break_dummies` | operational | One 0/1 dummy per date in `leaf_config.break_dates`; value is 1 from the break onward. |

### Functions & features

- Module: `macrocast.execution.deterministic` — `augment_frame(df, component, *, index=None, break_dates=None)` + `augment_array(X, component, *, index, break_dates=None)`.
- Wired into `_build_raw_panel_training_data` after preprocessing. Both X_train and X_pred are augmented identically so the fitted coefficients apply at prediction time.
- `monthly_seasonal` / `quarterly_seasonal` require a `DatetimeIndex`.
- Compile guard: `break_dummies` requires non-empty `leaf_config.break_dates`.

### Dropped values

- `trend_and_quadratic`: redundant with `linear_trend` + a future `leaf_config.trend_order` channel. The quadratic / higher-order polynomial trend will re-enter as a trend-order parameter when needed rather than a separate axis value.

### Recipe usage

```yaml
path:
  1_data_task:
    fixed_axes:
      deterministic_components: break_dummies
    leaf_config:
      break_dates: ["2008-09-01", "2020-03-01"]
  3_training:
    fixed_axes:
      feature_builder: raw_feature_panel
      model_family: ridge
```

---

## Benchmark & Predictor Universe (1.4) takeaways

- Every value in every 1.4 axis is operational in v1.0. Zero `registry_only` entries remain.
- `benchmark_family` gains 4 formerly-metadata variants as real implementations: `factor_model`, `multi_benchmark_suite`, `paper_specific_benchmark`, `survey_forecast`.
- `predictor_family` and `variable_universe` use the same design pattern: the user provides a pre-computed column list (or category mapping) via `leaf_config`; runtime discovery is out of scope.
- `deterministic_components` augments the raw-panel X with classical econometric terms (trend / seasonals / break dummies) via a dedicated module.

Next group: 1.5 Data handling policies (coming).
