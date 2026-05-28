# `correlation_view`

[Back to L1.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``correlation_view`` on sub-layer ``L1_5_E_correlation_pre_cleaning`` (layer ``l1_5``).

## Sub-layer

**L1_5_E_correlation_pre_cleaning**

## Axis metadata

- Default: `'none'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 4 option(s)
- Future: 0 option(s)

## Options

### `clustered_heatmap`  --  operational

Clustered heatmap with hierarchical reorder of rows and columns.

L1.5.E correlation visualisation ``clustered_heatmap``.

This option configures the ``correlation_view`` axis on the ``L1_5_E_correlation_pre_cleaning`` sub-layer of L1.5; output is emitted under ``manifest.diagnostics/l1_5/L1_5_E_correlation_pre_cleaning/`` alongside the other selected views.

**When to use**

Large panels where cluster structure aids reading; reveals correlated variable blocks.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`full_matrix`](#full-matrix), [`top_k_per_target`](#top-k-per-target), [`none`](#none)

_Last reviewed 2026-05-05 by macroforecast author._

### `full_matrix`  --  operational

Full N×N correlation matrix as a heatmap.

L1.5.E correlation visualisation ``full_matrix``.

This option configures the ``correlation_view`` axis on the ``L1_5_E_correlation_pre_cleaning`` sub-layer of L1.5; output is emitted under ``manifest.diagnostics/l1_5/L1_5_E_correlation_pre_cleaning/`` alongside the other selected views.

**When to use**

Small panels (N < 50) where every pairwise correlation fits on one figure.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`clustered_heatmap`](#clustered-heatmap), [`top_k_per_target`](#top-k-per-target), [`none`](#none)

_Last reviewed 2026-05-05 by macroforecast author._

### `none`  --  operational

Skip correlation diagnostics entirely.

L1.5.E correlation visualisation ``none``.

This option configures the ``correlation_view`` axis on the ``L1_5_E_correlation_pre_cleaning`` sub-layer of L1.5; output is emitted under ``manifest.diagnostics/l1_5/L1_5_E_correlation_pre_cleaning/`` alongside the other selected views.

**When to use**

Already covered by upstream EDA; reducing diagnostic surface.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`full_matrix`](#full-matrix), [`clustered_heatmap`](#clustered-heatmap), [`top_k_per_target`](#top-k-per-target)

_Last reviewed 2026-05-05 by macroforecast author._

### `top_k_per_target`  --  operational

Top-k highest-``|ρ|`` predictors per target.

L1.5.E correlation visualisation ``top_k_per_target``.

This option configures the ``correlation_view`` axis on the ``L1_5_E_correlation_pre_cleaning`` sub-layer of L1.5; output is emitted under ``manifest.diagnostics/l1_5/L1_5_E_correlation_pre_cleaning/`` alongside the other selected views.

**When to use**

Quickly identifying the most-correlated predictors when N is too large for a full matrix.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`full_matrix`](#full-matrix), [`clustered_heatmap`](#clustered-heatmap), [`none`](#none)

_Last reviewed 2026-05-05 by macroforecast author._
