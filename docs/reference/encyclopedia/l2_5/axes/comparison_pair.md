# `comparison_pair`

[Back to L2.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``comparison_pair`` on sub-layer ``L2_5_A_comparison_axis`` (layer ``l2_5``).

## Sub-layer

**L2_5_A_comparison_axis**

## Axis metadata

- Default: `'raw_vs_final_clean'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 5 option(s)
- Future: 0 option(s)

## Options

### `multi_stage`  --  operational

Compare every L2 stage in sequence (raw → tcoded → outlier → imputed → frame).

L2.5.A comparison stage selector ``multi_stage``. Defines which pair of panel snapshots feeds the comparison output forms below.

**When to use**

Stage-wise attribution of cleaning effects; diagnosing which stage is responsible for unexpected shifts.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`raw_vs_tcoded`](#raw-vs-tcoded), [`raw_vs_outlier_handled`](#raw-vs-outlier-handled), [`raw_vs_imputed`](#raw-vs-imputed), [`raw_vs_final_clean`](#raw-vs-final-clean)

_Last reviewed 2026-05-05 by macroforecast author._

### `raw_vs_final_clean`  --  operational

Compare raw panel vs end-of-L2 panel.

L2.5.A comparison stage selector ``raw_vs_final_clean``. Defines which pair of panel snapshots feeds the comparison output forms below.

**When to use**

Default; one-glance summary of L2's cumulative effect across all four stages.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`raw_vs_tcoded`](#raw-vs-tcoded), [`raw_vs_outlier_handled`](#raw-vs-outlier-handled), [`raw_vs_imputed`](#raw-vs-imputed), [`multi_stage`](#multi-stage)

_Last reviewed 2026-05-05 by macroforecast author._

### `raw_vs_imputed`  --  operational

Compare raw panel vs L2.D imputed panel.

L2.5.A comparison stage selector ``raw_vs_imputed``. Defines which pair of panel snapshots feeds the comparison output forms below.

**When to use**

Auditing the imputation method's footprint -- detecting bias introduced by EM-factor imputation.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`raw_vs_tcoded`](#raw-vs-tcoded), [`raw_vs_outlier_handled`](#raw-vs-outlier-handled), [`raw_vs_final_clean`](#raw-vs-final-clean), [`multi_stage`](#multi-stage)

_Last reviewed 2026-05-05 by macroforecast author._

### `raw_vs_outlier_handled`  --  operational

Compare raw panel vs L2.C outlier-handled panel.

L2.5.A comparison stage selector ``raw_vs_outlier_handled``. Defines which pair of panel snapshots feeds the comparison output forms below.

**When to use**

Auditing winsorisation / replacement effects on tail behaviour.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`raw_vs_tcoded`](#raw-vs-tcoded), [`raw_vs_imputed`](#raw-vs-imputed), [`raw_vs_final_clean`](#raw-vs-final-clean), [`multi_stage`](#multi-stage)

_Last reviewed 2026-05-05 by macroforecast author._

### `raw_vs_tcoded`  --  operational

Compare raw panel vs L2.B-transformed panel.

L2.5.A comparison stage selector ``raw_vs_tcoded``. Defines which pair of panel snapshots feeds the comparison output forms below.

**When to use**

Auditing the impact of tcode application -- e.g. confirming log_diff produced stationary series.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`raw_vs_outlier_handled`](#raw-vs-outlier-handled), [`raw_vs_imputed`](#raw-vs-imputed), [`raw_vs_final_clean`](#raw-vs-final-clean), [`multi_stage`](#multi-stage)

_Last reviewed 2026-05-05 by macroforecast author._
