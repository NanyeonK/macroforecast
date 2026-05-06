# `correlation_view`

[Back to L3.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``correlation_view`` on sub-layer ``L3_5_C_feature_correlation`` (layer ``l3_5``).

## Sub-layer

**L3_5_C_feature_correlation**

## Axis metadata

- Default: `'clustered_heatmap'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 3 option(s)
- Future: 0 option(s)

## Options

### `clustered_heatmap`  --  operational

Clustered heatmap reordered by hierarchical clustering.

L3.5.C correlation view ``clustered_heatmap``.

This option configures the ``correlation_view`` axis on the ``L3_5_C_feature_correlation`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_C_feature_correlation/`` alongside the other selected views.

**When to use**

Large feature panels with block structure; reveals clusters of correlated features.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`full_matrix`](#full-matrix), [`top_k`](#top-k)

_Last reviewed 2026-05-05 by macroforecast author._

### `full_matrix`  --  operational

Full feature × feature correlation matrix.

L3.5.C correlation view ``full_matrix``.

This option configures the ``correlation_view`` axis on the ``L3_5_C_feature_correlation`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_C_feature_correlation/`` alongside the other selected views.

**When to use**

Small feature panels (< 100 cols).

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`clustered_heatmap`](#clustered-heatmap), [`top_k`](#top-k)

_Last reviewed 2026-05-05 by macroforecast author._

### `top_k`  --  operational

Top-k highest-``|ρ|`` pairs.

L3.5.C correlation view ``top_k``.

This option configures the ``correlation_view`` axis on the ``L3_5_C_feature_correlation`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_C_feature_correlation/`` alongside the other selected views.

**When to use**

Cheapest readout for very wide panels.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`full_matrix`](#full-matrix), [`clustered_heatmap`](#clustered-heatmap)

_Last reviewed 2026-05-05 by macroforecast author._
