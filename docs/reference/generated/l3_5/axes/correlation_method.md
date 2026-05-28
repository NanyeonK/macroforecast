# `correlation_method`

[Back to L3.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``correlation_method`` on sub-layer ``L3_5_C_feature_correlation`` (layer ``l3_5``).

## Sub-layer

**L3_5_C_feature_correlation**

## Axis metadata

- Default: `'pearson'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 2 option(s)
- Future: 0 option(s)

## Options

### `pearson`  --  operational

Pearson correlation for feature pairs.

L3.5.C correlation method ``pearson``.

This option configures the ``correlation_method`` axis on the ``L3_5_C_feature_correlation`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_C_feature_correlation/`` alongside the other selected views.

**When to use**

Linear-association default. Activates the ``pearson`` branch on L3.5.correlation_method; combine with related options on the same sub-layer for a comprehensive diagnostic.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`spearman`](#spearman)

_Last reviewed 2026-05-05 by macroforecast author._

### `spearman`  --  operational

Spearman rank correlation.

L3.5.C correlation method ``spearman``.

This option configures the ``correlation_method`` axis on the ``L3_5_C_feature_correlation`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_C_feature_correlation/`` alongside the other selected views.

**When to use**

Monotonic, robust to outliers; preferred for non-Gaussian features.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`pearson`](#pearson)

_Last reviewed 2026-05-05 by macroforecast author._
