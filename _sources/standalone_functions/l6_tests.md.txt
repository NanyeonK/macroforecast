# Standalone functions — L6 statistical tests

L6 provides formal statistical tests for comparing forecasts. In the standalone
paradigm these are planned as:

```python
mf.functions.<test>(errors_a, errors_b, **kwargs) -> TestResult
```

A `TestResult` carries `.statistic`, `.p_value`, `.kernel`, `.lag`, and
`.reject_null` (at the configured alpha).

> **Cycle 22 note** — L6 standalone callables are planned for a future cycle.
> This page documents the 7 primary L6 test ops. The encyclopedia links point
> to full per-axis reference pages.

## Equal-predictive-ability tests (4 ops)

Tests for the null hypothesis that two forecasts have equal expected loss.

| Op | One-liner | Encyclopedia |
|---|---|---|
| `dm_diebold_mariano` | DM (1995) with Newey-West HAC + HLN small-sample correction | [equal_predictive_test axis](../encyclopedia/l6/axes/equal_predictive_test.md#dm-diebold-mariano) |
| `gw_giacomini_white` | GW (2006) conditional predictive ability test | [equal_predictive_test axis](../encyclopedia/l6/axes/equal_predictive_test.md#gw-giacomini-white) |
| `dmp_multi_horizon` | DMP joint multi-horizon HAC-adjusted test | [equal_predictive_test axis](../encyclopedia/l6/axes/equal_predictive_test.md#dmp-multi-horizon) |
| `harvey_newbold_encompassing` | HLN (1998) forecast encompassing test | [equal_predictive_test axis](../encyclopedia/l6/axes/equal_predictive_test.md#harvey-newbold-encompassing) |

**When to use equal-predictive tests**: Non-nested forecast comparisons where
you want a single p-value for MSE or MAE loss equality. Use DM by default;
switch to GW when you suspect regime-dependent performance differentials.

## Nested-model tests (3 ops)

Tests for the null hypothesis that a restricted (nested) model is adequate.
The Clark-West adjustment is required for nested-model comparisons because
the unrestricted model's additional noise inflates MSE in finite samples.

| Op | One-liner | Encyclopedia |
|---|---|---|
| `clark_west` | Clark-West (2007) MSE-adjusted t-test for nested models | [cw_adjustment axis](../encyclopedia/l6/axes/cw_adjustment.md) |
| `enc_new` | Harvey-Leybourne-Newbold (1998) ENC-NEW encompassing | [enc_test_one_sided axis](../encyclopedia/l6/axes/enc_test_one_sided.md) |
| `enc_t` | ENC-T variant (regression-based encompassing) | [enc_test_one_sided axis](../encyclopedia/l6/axes/enc_test_one_sided.md) |

**When to use nested-model tests**: Comparing a benchmark AR/RW against an
augmented model. The Clark-West test corrects for the downward MSE bias from
the extra noise in the augmented model's forecast.

## Quick example (recipe DSL)

```yaml
6_statistical_tests:
  enabled: true
  sub_layers:
    L6_A_equal_predictive:
      enabled: true
      fixed_axes:
        equal_predictive_test: dm_diebold_mariano
        loss_function: mse
        hln_correction: true
    L6_B_nested:
      enabled: true
      fixed_axes:
        cw_adjustment: clark_west
```

## Related

- [L5 metrics](l5_metrics.md) — compute the loss values that feed L6 tests.
- [Encyclopedia L6 index](../encyclopedia/l6/index.md) — full test surface
  including MCS / SPA / StepM bootstrap tests, residual battery, and density
  tests.
