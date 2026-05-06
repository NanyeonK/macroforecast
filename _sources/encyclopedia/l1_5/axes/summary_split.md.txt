# `summary_split`

[Back to L1.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``summary_split`` on sub-layer ``L1_5_B_univariate_summary`` (layer ``l1_5``).

## Sub-layer

**L1_5_B_univariate_summary**

## Axis metadata

- Default: `'full_sample'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 4 option(s)
- Future: 0 option(s)

## Options

### `full_sample`  --  operational

Compute summary metrics over the entire sample.

Splits the L1.5.B summary table along ``full_sample``. Multi-select supported -- choosing two splits stacks the resulting tables vertically with the split label as a leading column.

**When to use**

Default; baseline distributional view across all observations.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`per_decade`](#per-decade), [`per_regime`](#per-regime), [`pre_oos_only`](#pre-oos-only)

_Last reviewed 2026-05-05 by macroforecast author._

### `per_decade`  --  operational

Compute summary metrics on each calendar decade (1980s / 1990s / ...).

Splits the L1.5.B summary table along ``per_decade``. Multi-select supported -- choosing two splits stacks the resulting tables vertically with the split label as a leading column.

**When to use**

Detecting structural shifts in volatility or central tendency over multi-decade samples.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`full_sample`](#full-sample), [`per_regime`](#per-regime), [`pre_oos_only`](#pre-oos-only)

_Last reviewed 2026-05-05 by macroforecast author._

### `per_regime`  --  operational

Compute summary metrics on each L1.G regime slice.

Splits the L1.5.B summary table along ``per_regime``. Multi-select supported -- choosing two splits stacks the resulting tables vertically with the split label as a leading column.

**When to use**

Regime-conditional descriptive statistics; requires non-pooled L1.G regime configuration.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`full_sample`](#full-sample), [`per_decade`](#per-decade), [`pre_oos_only`](#pre-oos-only)

_Last reviewed 2026-05-05 by macroforecast author._

### `pre_oos_only`  --  operational

Restrict summaries to the pre-OOS training window.

Splits the L1.5.B summary table along ``pre_oos_only``. Multi-select supported -- choosing two splits stacks the resulting tables vertically with the split label as a leading column.

**When to use**

Avoiding look-ahead in summaries used to motivate L2 / L3 hyperparameter choices.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`full_sample`](#full-sample), [`per_decade`](#per-decade), [`per_regime`](#per-regime)

_Last reviewed 2026-05-05 by macroforecast author._
