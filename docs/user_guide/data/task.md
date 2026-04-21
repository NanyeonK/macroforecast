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

**Selects how h-step-ahead forecasts are produced.** Both values are operational in v1.0, but the two are tied to specific `feature_builder` choices — the pairing is what the current runtime actually implements, and the default is picked dynamically.

### Value catalog

| Value | Status | Valid `feature_builder` | What it does |
|---|---|---|---|
| `direct` | operational | `raw_feature_panel`, `factor_pca`, `factors_plus_AR` | Fit one model on `(X_t, y_{t+h})` pairs and predict once. The raw panel path already does this — target `y_{t+h}` is shifted h steps into the future at training time. |
| `iterated` | operational | `autoreg_lagged_target` | Fit a 1-step model on `(lags_t, y_{t+1})` pairs and recurse h times, appending each prediction to the lag history (`_recursive_predict_sklearn`). Default for autoreg recipes. |

### Dynamic default

The compiler picks the default based on `feature_builder`:

- `feature_builder = autoreg_lagged_target` → default `forecast_type = iterated`
- `feature_builder = raw_feature_panel` (and other panel variants) → default `forecast_type = direct`

### Compatibility guards (v1.0)

Cross combinations are not runtime-wired in v1.0 and are rejected at compile time:

- `forecast_type = iterated` + `feature_builder = raw_feature_panel` → `blocked_by_incompatibility` (would require exogenous X forecasting).
- `forecast_type = direct` + `feature_builder = autoreg_lagged_target` → `blocked_by_incompatibility` (the autoreg path is iterated by construction; a true direct autoreg executor is deferred).

### Functions & features

- `macrocast.execution.build._recursive_predict_sklearn(model, train, horizon, lag_order)` — the iterated 1-step executor used by every autoreg_lagged_target model family.
- `macrocast.execution.build._build_raw_panel_training_data(frame, target, horizon, start, origin, contract)` — builds the direct h-step target `y.iloc[start+horizon : origin+1]` used by the raw-panel executors.
- Compiler-side wiring: `macrocast.compiler.build._data_task_spec` reads `feature_builder` and derives the default `forecast_type`; the compatibility guards live alongside the other §1.2 guards in the compile function's blocked-reasons block.
- Manifest: `data_task_spec["forecast_type"]` records the value (default or explicit) used for the run.

### Dropped values

`dirrec` (Taieb-Bontempi 2011 hybrid), `mimo`, `seq2seq` — niche / deep-only strategies that belong to model capabilities rather than to a general forecast_type. See coverage_ledger §1.3.2.

### Recipe usage

```yaml
# Autoreg recipe — iterated is the default; explicit is equivalent.
path:
  1_data_task:
    fixed_axes:
      forecast_type: iterated        # optional (default for autoreg)
  3_training:
    fixed_axes:
      feature_builder: autoreg_lagged_target
      model_family: ridge
```

```yaml
# Raw-panel recipe — direct is the default.
path:
  3_training:
    fixed_axes:
      feature_builder: raw_feature_panel
      model_family: lasso
```

---

## 1.2.3 `forecast_object`

**Selects what statistic of the predictive distribution the model emits.** All three values are operational in v1.0.

### Value catalog

| Value | Status | What it does |
|---|---|---|
| `point_mean` | operational | Default. Every executor emits a conditional-mean point forecast; MSFE / RMSE metrics are computed on this. |
| `point_median` | operational | Conditional-median point forecast. Compatible with `model_family=quantile_linear` (which fits a quantile regression at `τ=0.5`). |
| `quantile` | operational | Quantile forecast at a user-specified `τ`. Requires `model_family=quantile_linear`; the level comes from `training_spec.hp.quantile` (default `0.5`). |

### Compatibility guard (v1.0)

The v1.0 guard enforces `model_family=quantile_linear ⇒ forecast_object ∈ {point_median, quantile}`. Setting `forecast_object=point_mean` with `quantile_linear` is rejected at compile time.

### Picking a quantile level

```yaml
# Upper-tail forecast at tau = 0.9
path:
  1_data_task:
    fixed_axes:
      forecast_object: quantile
  3_training:
    fixed_axes:
      model_family: quantile_linear
    leaf_config:
      hp:
        quantile: 0.9
```

If `quantile` is set but no explicit `hp.quantile` is provided, the underlying `QuantileRegressor` falls back to the library default `τ = 0.5` — numerically the same forecast as `point_median`. Pick the level explicitly when you want a non-median quantile.

### Functions & features

- `macrocast.execution.deep_training._build_model("quantile_linear", hp)` wraps `sklearn.linear_model.QuantileRegressor(quantile=hp.get("quantile", 0.5), alpha=hp.get("alpha", 1.0), solver="highs")`. The `quantile` hyperparameter is the τ level applied at fit time.
- Compiler guard lives in `macrocast.compiler.build`'s main compile function and enforces `model_family=quantile_linear ⇒ forecast_object ∈ {point_median, quantile}`.
- Manifest: `data_task_spec["forecast_object"]` records the selected value; the τ level (if provided) is carried through `training_spec["hp"]["quantile"]`.

### Dropped values

- `direction`: sign of the point forecast; this is a metric view (Pesaran-Timmermann, binomial-hit), not an independent forecast object.
- `interval`: subsumed by the v1.0 conformal wrapper on the point forecast (future work — not a separate axis).
- `density`: v2 distributional work.
- `turning_point`, `regime_probability`, `event_probability`: niche / bound to state_space / event modules that do not exist in v1.x.

### Recipe usage

```yaml
# Default conditional-mean point forecast
path:
  1_data_task:
    fixed_axes:
      forecast_object: point_mean
```

---

## 1.2.4 `horizon_target_construction`

**Selects the metric scale on which forecasts are evaluated.** All four values are operational in v1.0.

### Value catalog

| Value | Status | What it does |
|---|---|---|
| `future_level_y_t_plus_h` | operational | Default. Metrics on the raw y-level: `error = y_{t+h} - ŷ_{t+h}`. |
| `future_diff` | operational | Metrics on 1st-difference scale: `y_true = y_{t+h} - y_t`, `ŷ = ŷ_{t+h} - y_t`. Error equals level-scale error (anchor cancels), but reported y_true / y_pred move onto the diff scale. |
| `future_logdiff` | operational | Metrics on log-growth scale: `y_true = log(y_{t+h}) - log(y_t)`, `ŷ = log(ŷ_{t+h}) - log(y_t)`. Error = `log(y_{t+h}/ŷ_{t+h})`, a relative / percentage-style residual. Requires strictly positive y. |

### Semantics (v1.0)

v1.0 implements `horizon_target_construction` as a **metric-scale transform** at the central row-computation site (`_compute_origin` in `execution.build`). The model executor always emits a level-scale forecast `ŷ_{t+h}`. Immediately before the row is assembled, both the forecast and the realised `y_{t+h}` are forward-transformed using `y_anchor = y_t` (the last observed level at the forecast origin), so `error` / `abs_error` / `squared_error` — and the corresponding benchmark fields — land on the construction scale. Level-scale values are preserved as `y_true_level` / `y_pred_level` / `benchmark_pred_level` on every row for provenance.

Training-time target transforms (fit the model directly on the transformed y, with an inverse mapping from prediction back to level) are **not** what v1.0 does — those land in the Layer-2 `target_transform` work and remain on the v1.1 roadmap for bidirectional wiring.

### Functions & features

- `macrocast.execution.horizon_target.forward_scalar(y_val, y_anchor, construction)` — applies the selected construction to a single scalar.
- `build_horizon_target(y, horizon, construction)` / `inverse_horizon_target(y_hat, y_anchor, construction)` are exported for future training-time use but are not yet wired in v1.0.
- Provenance columns added to every prediction row: `horizon_target_construction`, `y_true_level`, `y_pred_level`, `benchmark_pred_level`.

### Dropped values

`annualized_growth_to_h` is a linear (×12/h) transform of `cumulative_growth_to_h` — it belongs in metric-time reporting, not as a distinct target shape. `average_growth_1_to_h` is a scaled variant of cumulative. `realized_future_average`, `future_sum` are niche. `future_indicator` overlapped the dropped `forecast_object=direction / event_probability`.

### Recipe usage

```yaml
# Default: level-scale metrics
path:
  1_data_task:
    fixed_axes:
      horizon_target_construction: future_level_y_t_plus_h
```

```yaml
# Log-growth-rate evaluation (CLSS 2021 style)
path:
  1_data_task:
    fixed_axes:
      horizon_target_construction: future_logdiff
```

---

## Task & Target (1.2) takeaways

- **`task`** is the only §1.2 axis that truly branches at runtime today. It flows into multi-target aggregator activation and `experiment_unit` default derivation.
- **`forecast_type`** is feature-builder-dynamic: `iterated` for `autoreg_lagged_target`, `direct` for `raw_feature_panel` and the panel variants. Cross combinations are blocked at compile time.
- **`forecast_object`** has all three values operational (`point_mean`, `point_median`, `quantile`). `quantile` pairs with `model_family=quantile_linear`; level via `hp.quantile`.
- **`horizon_target_construction`** is fully operational with 3 values — default `future_level_y_t_plus_h` plus 2 metric-scale transforms (`future_diff`, `future_logdiff`). `cumulative_growth_to_h` was dropped as a duplicate of future_logdiff (identical telescoping-sum formula).
- `target_family`, `multi_target_architecture`, `target_to_target_inclusion` are gone — see the "Dropped axes" note at the top.

Next group: §1.3 Horizon & evaluation window (coming).
