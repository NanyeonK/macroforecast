# `output_table_format`

[Back to L7](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``output_table_format`` on sub-layer ``L7_B_output_shape_export`` (layer ``l7``).

## Sub-layer

**L7_B_output_shape_export**

## Axis metadata

- Default: `'long'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 2 option(s)
- Future: 0 option(s)

## Options

### `long`  --  operational

Long-form (tidy) tables: one row per (model, feature, metric).

Returns importance tables in the tidy data format -- each row is a single observation of (model_id, feature, metric_value). Default for downstream pandas / R analysis since aggregation, filtering, and ggplot-style faceting all expect this shape.

Wickham's tidy-data principles (one variable per column, one observation per row, one type per table) underpin the long format.

**When to use**

Default for downstream pandas / R analysis; required for ``seaborn`` faceting.

**When NOT to use**

Paper-quality matrix-shaped reporting (use ``wide`` instead).

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Wickham (2014) 'Tidy Data', Journal of Statistical Software 59(10): 1-23. (doi:10.18637/jss.v059.i10)

**Related options**: [`wide`](#wide)

_Last reviewed 2026-05-05 by macroforecast author._

### `wide`  --  operational

Wide-form tables: one row per feature, columns per (model, metric).

Returns importance tables in the matrix-shaped format -- each row is one feature, columns vary across (model_id × metric) combinations. Compact for paper-quality reporting and the natural shape for LaTeX ``tabular`` export.

**When to use**

Compact paper-quality reporting; LaTeX table generation.

**When NOT to use**

Downstream pandas analysis -- use ``long`` instead.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'

**Related options**: [`long`](#long)

_Last reviewed 2026-05-05 by macroforecast author._
