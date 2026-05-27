# `distribution_view`

[Back to L2.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``distribution_view`` on sub-layer ``L2_5_B_distribution_shift`` (layer ``l2_5``).

## Sub-layer

**L2_5_B_distribution_shift**

## Axis metadata

- Default: `'multi'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 4 option(s)
- Future: 0 option(s)

## Options

### `histogram_overlay`  --  operational

Overlaid histograms (raw vs cleaned).

L2.5.B distribution view ``histogram_overlay``.

This option configures the ``distribution_view`` axis on the ``L2_5_B_distribution_shift`` sub-layer of L2.5; output is emitted under ``manifest.diagnostics/l2_5/L2_5_B_distribution_shift/`` alongside the other selected views.

**When to use**

Eyeballing distribution-shift magnitude; the most intuitive view for sharing with non-technical stakeholders.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`qq_plot`](#qq-plot), [`summary_table`](#summary-table), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `multi`  --  operational

Produce all three views together.

L2.5.B distribution view ``multi``.

This option configures the ``distribution_view`` axis on the ``L2_5_B_distribution_shift`` sub-layer of L2.5; output is emitted under ``manifest.diagnostics/l2_5/L2_5_B_distribution_shift/`` alongside the other selected views.

**When to use**

Default rich diagnostic for distribution-shift audits.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`histogram_overlay`](#histogram-overlay), [`qq_plot`](#qq-plot), [`summary_table`](#summary-table)

_Last reviewed 2026-05-05 by macroforecast author._

### `qq_plot`  --  operational

Q-Q plot of cleaned vs raw quantiles.

L2.5.B distribution view ``qq_plot``.

This option configures the ``distribution_view`` axis on the ``L2_5_B_distribution_shift`` sub-layer of L2.5; output is emitted under ``manifest.diagnostics/l2_5/L2_5_B_distribution_shift/`` alongside the other selected views.

**When to use**

Detecting tail-deformation patterns; deviations from the 45° line localise where cleaning altered the distribution.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`histogram_overlay`](#histogram-overlay), [`summary_table`](#summary-table), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `summary_table`  --  operational

Tabular distribution-metric summary.

L2.5.B distribution view ``summary_table``.

This option configures the ``distribution_view`` axis on the ``L2_5_B_distribution_shift`` sub-layer of L2.5; output is emitted under ``manifest.diagnostics/l2_5/L2_5_B_distribution_shift/`` alongside the other selected views.

**When to use**

Compact numeric summary per series; pairs naturally with ``distribution_metric`` choices.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`histogram_overlay`](#histogram-overlay), [`qq_plot`](#qq-plot), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._
