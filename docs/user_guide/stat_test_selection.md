# Choosing a Statistical Test Axis (Phase 2+)

Before Phase 2, macrocast had a single flat axis `stat_test` with 20 entries. In v0.4 we split it into 8 semantic axes so a recipe declares *what question it is asking*, not just which test name to run.

This page is a decision tree. Pick one axis per question you want answered; leave the others as `none` or omit them entirely.

## Step 1: what kind of comparison?

| If you are asking ... | Use axis | Operational values |
|---|---|---|
| "Do two non-nested models have equal predictive accuracy?" | `equal_predictive` | `dm`, `dm_hln`, `dm_modified` |
| "Does a nested model beat its parent?" | `nested` | `cw`, `enc_new`, `mse_f`, `mse_t` |
| "Is my model stable over time? Does it fail in subsamples?" | `cpa_instability` | `cpa`, `rossi`, `rolling_dm` |
| "Is my model in the top-k across many candidates?" | `multiple_model` | `reality_check`, `spa`, `mcs` |
| "Are my density / interval forecasts calibrated?" | `density_interval` | (Phase 10 §10.8 — none operational yet) |
| "Does my model predict direction better than chance?" | `direction` | `pesaran_timmermann`, `binomial_hit` |
| "Are residuals well-behaved?" | `residual_diagnostics` | `mincer_zarnowitz`, `ljung_box`, `arch_lm`, `bias_test`, `diagnostics_full` |
| "How should tests be sliced across targets / horizons / model pairs?" | `test_scope` | `per_target`, `per_horizon`, `per_model_pair` |

Multiple axes can be set on the same recipe — the runner dispatches each independently and bundles results in `stat_tests.json`.

## Step 2: pick the specific test

Each axis's operational values live in `macrocast/registry/tests/<axis>.py`. The status field distinguishes `operational` (runs today) from `planned` (deferred to v1.1 via Phase 10 §10.8). A recipe that names a `planned` value receives a `NotImplementedError` entry in its axis result, not a hard failure.

## Step 3: legacy `stat_test` (deprecation window)

Recipes that still set `stat_test: dm` keep working through v1.1 — a migration shim rewrites them into the equivalent 8-axis form (`equal_predictive: dm`) and emits a `DeprecationWarning`. The legacy field is scheduled for removal in v1.2 (ADR-006 breaking-change window).

To silence the warning, migrate your recipes:

```diff
 6_stat_tests:
   fixed_axes:
-    stat_test: dm
+    equal_predictive: dm
```

## Step 4: multi-axis sweeps

Each of the 8 axes can appear under a layer's `sweep_axes` block, so a horse-race recipe can compare, say, DM against Clark-West as a two-variant sweep:

```yaml
6_stat_tests:
  fixed_axes:
    test_scope: per_horizon
  sweep_axes:
    # Variants: {equal_predictive: dm} and {nested: cw}
    # Note: each sweep value still lives under one axis key, not two.
```

See `sweep_recipes.md` for the full grammar.

## See also

- [stat_tests.md](stat_tests.md) — per-axis reference
- [sweep_recipes.md](sweep_recipes.md) — sweep grammar
- [Mathematical definitions](../math/stat_tests.md)
