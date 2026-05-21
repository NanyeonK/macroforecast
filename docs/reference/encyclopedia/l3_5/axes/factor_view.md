# `factor_view`

[Back to L3.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``factor_view`` on sub-layer ``L3_5_B_factor_block_inspection`` (layer ``l3_5``).

## Sub-layer

**L3_5_B_factor_block_inspection**

## Axis metadata

- Default: `'multi'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 5 option(s)
- Future: 0 option(s)

## Options

### `cumulative_variance`  --  operational

Cumulative explained-variance curve.

L3.5.B factor view ``cumulative_variance``.

This option configures the ``factor_view`` axis on the ``L3_5_B_factor_block_inspection`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_B_factor_block_inspection/`` alongside the other selected views.

**When to use**

Quantifying how much variance the chosen ``n_components`` retains; threshold heuristics (80% / 90%) live here.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'
* Stock & Watson (2002) 'Forecasting Using Principal Components from a Large Number of Predictors', JASA 97(460): 1167-1179.

**Related options**: [`scree_plot`](#scree-plot), [`loadings_heatmap`](#loadings-heatmap), [`factor_timeseries`](#factor-timeseries), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `factor_timeseries`  --  operational

Estimated factor time-series plot.

L3.5.B factor view ``factor_timeseries``.

This option configures the ``factor_view`` axis on the ``L3_5_B_factor_block_inspection`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_B_factor_block_inspection/`` alongside the other selected views.

**When to use**

Confirming factors track recognisable cycles (NBER recessions, oil-price spikes, etc.).

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'
* Stock & Watson (2002) 'Forecasting Using Principal Components from a Large Number of Predictors', JASA 97(460): 1167-1179.

**Related options**: [`scree_plot`](#scree-plot), [`loadings_heatmap`](#loadings-heatmap), [`cumulative_variance`](#cumulative-variance), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `loadings_heatmap`  --  operational

Heatmap of factor loadings (factors × predictors).

L3.5.B factor view ``loadings_heatmap``.

This option configures the ``factor_view`` axis on the ``L3_5_B_factor_block_inspection`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_B_factor_block_inspection/`` alongside the other selected views.

**When to use**

Interpreting factor identity; high-loading variables suggest the factor's economic interpretation.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'
* Stock & Watson (2002) 'Forecasting Using Principal Components from a Large Number of Predictors', JASA 97(460): 1167-1179.

**Related options**: [`scree_plot`](#scree-plot), [`factor_timeseries`](#factor-timeseries), [`cumulative_variance`](#cumulative-variance), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `multi`  --  operational

Render every factor view together.

L3.5.B factor view ``multi``.

This option configures the ``factor_view`` axis on the ``L3_5_B_factor_block_inspection`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_B_factor_block_inspection/`` alongside the other selected views.

**When to use**

Default rich factor diagnostic; the standard package for factor-model papers.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'
* Stock & Watson (2002) 'Forecasting Using Principal Components from a Large Number of Predictors', JASA 97(460): 1167-1179.

**Related options**: [`scree_plot`](#scree-plot), [`loadings_heatmap`](#loadings-heatmap), [`factor_timeseries`](#factor-timeseries), [`cumulative_variance`](#cumulative-variance)

_Last reviewed 2026-05-05 by macroforecast author._

### `scree_plot`  --  operational

Eigenvalue scree plot for PCA / SPCA / DFM blocks.

L3.5.B factor view ``scree_plot``.

This option configures the ``factor_view`` axis on the ``L3_5_B_factor_block_inspection`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_B_factor_block_inspection/`` alongside the other selected views.

**When to use**

Choosing ``n_components`` -- the elbow heuristic remains the most popular tool.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'
* Stock & Watson (2002) 'Forecasting Using Principal Components from a Large Number of Predictors', JASA 97(460): 1167-1179.

**Related options**: [`loadings_heatmap`](#loadings-heatmap), [`factor_timeseries`](#factor-timeseries), [`cumulative_variance`](#cumulative-variance), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._
