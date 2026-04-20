# Task & Target (1.2)

Declares **what is being forecast and how the prediction is framed**. These four axes together answer: single- or multi-target? direct or iterated multi-step? what forecast object (point / quantile / ...)? and how is y_{t+h} constructed from the raw series?

| § | axis | Role |
|---|---|---|
| 1.2.1 | [`task`](#121-task) | Single-target vs. multi-target point forecasting |
| 1.2.2 | [`forecast_type`](#122-forecast_type) | Direct h-step vs. iterated 1-step |
| 1.2.3 | [`forecast_object`](#123-forecast_object) | Point (mean / median) vs. quantile — i.e. what statistic of the predictive distribution is returned |
| 1.2.4 | [`horizon_target_construction`](#124-horizon_target_construction) | How `y_{t+h}` is constructed from the raw series (level vs. growth variants) |

**Note on dropped axes:**

- `target_family` was dropped (PR #32) — its two operational values (`single_macro_series` / `multiple_macro_series`) duplicated `task`. Future panel / state / factor / latent / constructed / classification targets will re-enter as independent axes in v1.1+ when their runtime arrives.
- `multi_target_architecture` was dropped (PR #32) — duplicated `experiment_unit` (§0.3), which already owns separate_runs / shared_design dispatch.
- `target_to_target_inclusion` was dropped (this cleanup) — operational set was a single hardcoded value with no dispatch. If cross-target predictor policy becomes a real choice in v1.1, it will re-enter as a clean axis.

---

## 1.2.1 `task`

**Selects the overall forecasting task.** Every recipe picks exactly one.

### Value catalog

| Value | Status | What it does |
|---|---|---|
| `single_target_point_forecast` | operational | One target variable. Standard univariate-output benchmarking — this is the path every executor, metric, and statistical test in v1.0 is written for. |
| `multi_target_point_forecast` | operational | Multiple target variables. Triggers the multi-target aggregator in `execute_recipe`; runner choice (`separate_runs` vs. `shared_design`) is owned by `experiment_unit` (§0.3). |

### Functions & features

- Compiler branches at three sites in `compiler/build.py`: default derivation for dependent axes (line 397), multi-target aggregator activation (line 516), `experiment_unit` compatibility guard (line 542), and downstream spec propagation (line 747).
- `derive_experiment_unit_default` (`macrocast.design`) reads `task` to pick between single-target units (`single_target_single_model` / `single_target_model_grid` / `single_target_full_sweep`) and `multi_target_shared_design` as the default recipe shape.
- No standalone module — the task identity flows through `CompiledRecipeSpec.task`.

### Recipe usage

```yaml
path:
  1_data_task:
    fixed_axes:
      task: multi_target_point_forecast
    leaf_config:
      target: [INDPRO, UNRATE, CPIAUCSL]
      horizons: [1, 3, 6]
```

---

## 1.2.2 `forecast_type`

**Selects how h-step-ahead forecasts are produced.** Only one value is runtime-wired in v1.0.

### Value catalog

| Value | Status | What it does |
|---|---|---|
| `direct` | operational | Fit one model per horizon on `(X_t, y_{t+h})` pairs. This is what every executor in v1.0 does — sklearn / statsmodels / deep / AR all train direct-per-h. |
| `iterated` | registry_only (v1.1) | Fit a 1-step model and roll forward h times. **Not implemented in v1.0.** Recipes that set this value compile to `execution_status=representable_but_not_executable`. |

### v1.1 commitment

`iterated` will get a dedicated wrapper that runs any direct-1-step model recursively. Comparison against `direct` is the Marcellino-Stock-Watson (2006) classic — keeping `iterated` in the registry makes that comparison a first-class sweep axis.

### Dropped values

`dirrec` (niche hybrid), `mimo`, `seq2seq` (deep-only strategies that belong to model capabilities rather than to a general forecast_type) — see coverage_ledger §1.3.2.

### Recipe usage

```yaml
# v1.0 — only executable form
path:
  1_data_task:
    fixed_axes:
      forecast_type: direct
```

---

## 1.2.3 `forecast_object`

**Selects what statistic of the predictive distribution the model emits.** Two operational values in v1.0.

### Value catalog

| Value | Status | What it does |
|---|---|---|
| `point_mean` | operational | Default. Every executor emits a conditional-mean point forecast; MSFE / RMSE metrics are computed on this. |
| `point_median` | operational | Conditional-median point forecast. `model_family=quantile_linear` is gated by the compiler (line 641) to require this value. |
| `quantile` | registry_only (v1.1) | Quantile forecast at declared level(s). **Not implemented in v1.0.** |

### v1.1 commitment

`quantile` will be unlocked together with a conformal / quantile-loss pipeline. Prediction-interval output emerges as a by-product — there is no separate `interval` axis value.

### Dropped values

- `direction`: sign of the point forecast; this is a metric view (Pesaran-Timmermann, binomial-hit), not an independent forecast object.
- `interval`: subsumed by v1.1 conformal wrapper on the point forecast.
- `density`: v2 distributional work.
- `turning_point`, `regime_probability`, `event_probability`: niche / bound to state_space / event modules that do not exist in v1.x.

### Recipe usage

```yaml
# v1.0 executable forms
path:
  1_data_task:
    fixed_axes:
      forecast_object: point_mean       # default
      # or: forecast_object: point_median  (required with quantile_linear model)
```

---

## 1.2.4 `horizon_target_construction`

**Selects how `y_{t+h}` is constructed from the raw series.** One operational value in v1.0; three v1.1 commitments for the target-transform inverse pipeline.

### Value catalog

| Value | Status | What it does |
|---|---|---|
| `future_level_y_t_plus_h` | operational | `y_{t+h}` is the raw level — `y.shift(-h)`. Metrics are on the same scale as the input series. |
| `future_diff` | registry_only (v1.1) | `y_{t+h} - y_t`. Requires inverse-transform to return forecasts and metrics on the original level. |
| `future_logdiff` | registry_only (v1.1) | `log(y_{t+h}) - log(y_t)` — log-growth. |
| `cumulative_growth_to_h` | registry_only (v1.1) | `log(y_{t+h}) - log(y_t)` aggregated over the h-step window (CLSS 2021 style). |

### v1.1 commitment

The three non-`future_level_y_t_plus_h` values share one v1.1 deliverable: a target-transform pipeline with real forward + inverse mapping so metrics land on the raw scale. Today's `target_transform` axis (Layer 2) is forward-only — metrics come out on the transformed scale, which is a known v0.9.2 limitation.

### Dropped values

`annualized_growth_to_h` is a linear (×12/h) transform of `cumulative_growth_to_h` — it belongs in metric-time reporting, not as a distinct target shape. `average_growth_1_to_h` is a scaled variant of cumulative. `realized_future_average`, `future_sum` are niche. `future_indicator` overlapped the dropped `forecast_object=direction / event_probability`.

### Recipe usage

```yaml
# v1.0 — only executable form
path:
  1_data_task:
    fixed_axes:
      horizon_target_construction: future_level_y_t_plus_h
```

---

## Task & Target (1.2) takeaways

- **`task`** is the only §1.2 axis that truly branches at runtime today. It flows into multi-target aggregator activation and `experiment_unit` default derivation.
- **`forecast_type`**, **`forecast_object`**, **`horizon_target_construction`** each have exactly one executable value in v1.0. Every other value survives as `registry_only` with an explicit v1.1 runtime commitment, or was dropped.
- **Five registry_only values** constitute the v1.1 §1.2 roadmap: `iterated`, `quantile`, `future_diff`, `future_logdiff`, `cumulative_growth_to_h`. They are the acceptance criteria for §1.2 completeness in phase-10.
- `target_family`, `multi_target_architecture`, `target_to_target_inclusion` are gone — see the "Dropped axes" note at the top.

Next group: §1.3 Horizon & evaluation window (coming).
