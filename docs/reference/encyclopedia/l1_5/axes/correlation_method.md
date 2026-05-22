# `correlation_method`

[Back to L1.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``correlation_method`` on sub-layer ``L1_5_E_correlation_pre_cleaning`` (layer ``l1_5``).

## Sub-layer

**L1_5_E_correlation_pre_cleaning**

## Axis metadata

- Default: `'pearson'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 3 option(s)
- Future: 0 option(s)

## Options

### `kendall`  --  operational

Kendall tau rank correlation.

L1.5.E correlation method ``kendall``.

This option configures the ``correlation_method`` axis on the ``L1_5_E_correlation_pre_cleaning`` sub-layer of L1.5; output is emitted under ``manifest.diagnostics/l1_5/L1_5_E_correlation_pre_cleaning/`` alongside the other selected views.

**When to use**

Conservative rank measure; smaller variance than Spearman in small samples (n < 30).

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`pearson`](#pearson), [`spearman`](#spearman)

_Last reviewed 2026-05-05 by macroforecast author._

### `pearson`  --  operational

Pearson product-moment correlation -- linear association measure.

L1.5.E correlation method ``pearson``.

This option configures the ``correlation_method`` axis on the ``L1_5_E_correlation_pre_cleaning`` sub-layer of L1.5; output is emitted under ``manifest.diagnostics/l1_5/L1_5_E_correlation_pre_cleaning/`` alongside the other selected views.

**When to use**

Default; assumes approximate normality of pairs and linear association.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`spearman`](#spearman), [`kendall`](#kendall)

_Last reviewed 2026-05-05 by macroforecast author._

### `spearman`  --  operational

Spearman rank correlation -- monotonic-association measure.

L1.5.E correlation method ``spearman``.

This option configures the ``correlation_method`` axis on the ``L1_5_E_correlation_pre_cleaning`` sub-layer of L1.5; output is emitted under ``manifest.diagnostics/l1_5/L1_5_E_correlation_pre_cleaning/`` alongside the other selected views.

**When to use**

Robust to outliers and non-normal marginals; preferred when pairs have heavy tails.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`pearson`](#pearson), [`kendall`](#kendall)

_Last reviewed 2026-05-05 by macroforecast author._
