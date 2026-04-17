# Statistical Tests

Layer 6 of a macrocast recipe selects which hypothesis tests run against the forecast artifact. Since v0.4 (Phase 2), this layer is organised as **8 semantic axes** instead of the single flat `stat_test` axis.

## The 8 axes

Each axis lives in `macrocast/registry/tests/<axis>.py` and follows the standard `AxisDefinition` pattern. All axes default to `none`.

| Axis | Purpose |
|------|---------|
| `equal_predictive` | Non-nested loss-differential tests |
| `nested` | Nested-model forecast accuracy tests |
| `cpa_instability` | Conditional predictive ability + stability / break tests |
| `multiple_model` | Multi-model superiority / confidence sets |
| `density_interval` | Density-forecast + interval-coverage tests |
| `direction` | Directional-accuracy tests |
| `residual_diagnostics` | Residual regression / serial / heteroskedasticity diagnostics |
| `test_scope` | Meta-axis controlling where tests are applied (per-target, per-horizon, per-model-pair) |

See [stat_test_selection.md](stat_test_selection.md) for the decision tree.

## Operational values (v0.4)

### equal_predictive
- `dm` — Diebold-Mariano
- `dm_hln` — DM with Harvey-Leybourne-Newbold small-sample correction
- `dm_modified` — Modified DM for long-horizon forecasts

### nested
- `cw` — Clark-West
- `enc_new` — ENC-NEW forecast encompassing
- `mse_f` — MSE-F statistic
- `mse_t` — MSE-t statistic

### cpa_instability
- `cpa` — Giacomini-White conditional predictive ability
- `rossi` — Rossi-Sekhposyan stability statistic
- `rolling_dm` — Rolling-window DM summary

### multiple_model
- `reality_check` — White Reality Check bootstrap
- `spa` — Hansen SPA bootstrap
- `mcs` — Model Confidence Set

### direction
- `pesaran_timmermann` — Directional-accuracy test
- `binomial_hit` — Binomial hit-rate test

### residual_diagnostics
- `mincer_zarnowitz` — Mincer-Zarnowitz regression
- `ljung_box` — Serial correlation
- `arch_lm` — ARCH-LM heteroskedasticity
- `bias_test` — Forecast-bias t-test
- `diagnostics_full` — Residual diagnostic bundle

### test_scope
- `per_target` — tests run per target
- `per_horizon` — tests run per (target, horizon)
- `per_model_pair` — pairwise tests across all model pairs

### density_interval

No operational values in v0.4. The axis is registered with 7 `planned` values (PIT uniformity, Berkowitz, Kupiec, Christoffersen family, interval coverage) that will land via Phase 10 §10.8 (v1.1).

## Output layout

`execute_recipe` writes one `stat_tests.json` bundle at the study root, keyed by axis, with each axis payload carrying the usual `stat_test`, `statistic`, `p_value`, `n` keys. For backwards compatibility during the v1.x window, successful single-test axes also receive the pre-v0.4 per-test file (`stat_test_<test_value>.json`).

## Recipe example

```yaml
6_stat_tests:
  fixed_axes:
    equal_predictive: dm_hln
    nested: cw
    residual_diagnostics: ljung_box
    test_scope: per_horizon
```

This runs 3 tests (+ meta-axis recording) and writes `stat_tests.json` plus `stat_test_dm_hln.json`, `stat_test_cw.json`, `stat_test_ljung_box.json`.

## Legacy `stat_test` field

Still honored through v1.1; emits `DeprecationWarning` when used. The migration shim lives in `macrocast/compiler/migrations/stat_test_split.py`. Removal is scheduled for v1.2 (ADR-006).
