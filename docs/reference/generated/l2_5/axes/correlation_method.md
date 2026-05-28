# `correlation_method`

[Back to L2.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``correlation_method`` on sub-layer ``L2_5_C_correlation_shift`` (layer ``l2_5``).

## Sub-layer

**L2_5_C_correlation_shift**

## Axis metadata

- Default: `'pearson'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 2 option(s)
- Future: 0 option(s)

## Options

### `pearson`  --  operational

Pearson correlation for the shift comparison.

L2.5.C correlation method ``pearson``.

This option configures the ``correlation_method`` axis on the ``L2_5_C_correlation_shift`` sub-layer of L2.5; output is emitted under ``manifest.diagnostics/l2_5/L2_5_C_correlation_shift/`` alongside the other selected views.

**When to use**

Default linear-association measure; easiest to interpret.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`spearman`](#spearman)

_Last reviewed 2026-05-05 by macroforecast author._

### `spearman`  --  operational

Spearman rank correlation.

L2.5.C correlation method ``spearman``.

This option configures the ``correlation_method`` axis on the ``L2_5_C_correlation_shift`` sub-layer of L2.5; output is emitted under ``manifest.diagnostics/l2_5/L2_5_C_correlation_shift/`` alongside the other selected views.

**When to use**

Robust to outliers; preferred when L2.C may have changed tail behaviour.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`pearson`](#pearson)

_Last reviewed 2026-05-05 by macroforecast author._
