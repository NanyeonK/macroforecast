# `comparison_output_form`

[Back to L2.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``comparison_output_form`` on sub-layer ``L2_5_A_comparison_axis`` (layer ``l2_5``).

## Sub-layer

**L2_5_A_comparison_axis**

## Axis metadata

- Default: `'multi'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 5 option(s)
- Future: 0 option(s)

## Options

### `difference_table`  --  operational

Table of (after - before) per metric per series.

L2.5.A comparison output form ``difference_table``.

This option configures the ``comparison_output_form`` axis on the ``L2_5_A_comparison_axis`` sub-layer of L2.5; output is emitted under ``manifest.diagnostics/l2_5/L2_5_A_comparison_axis/`` alongside the other selected views.

**When to use**

Quantifying the magnitude of cleaning shifts; ordered by absolute change for easy ranking.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`side_by_side_summary`](#side-by-side-summary), [`overlay_timeseries`](#overlay-timeseries), [`distribution_shift`](#distribution-shift), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `distribution_shift`  --  operational

Distribution-overlap statistics (KS / histogram).

L2.5.A comparison output form ``distribution_shift``.

This option configures the ``comparison_output_form`` axis on the ``L2_5_A_comparison_axis`` sub-layer of L2.5; output is emitted under ``manifest.diagnostics/l2_5/L2_5_A_comparison_axis/`` alongside the other selected views.

**When to use**

Detecting cleaning steps that materially change distributions, not just summary statistics.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`side_by_side_summary`](#side-by-side-summary), [`difference_table`](#difference-table), [`overlay_timeseries`](#overlay-timeseries), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `multi`  --  operational

Render every comparison output form together.

L2.5.A comparison output form ``multi``.

This option configures the ``comparison_output_form`` axis on the ``L2_5_A_comparison_axis`` sub-layer of L2.5; output is emitted under ``manifest.diagnostics/l2_5/L2_5_A_comparison_axis/`` alongside the other selected views.

**When to use**

Comprehensive cleaning audit; recommended default for first-time runs.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`side_by_side_summary`](#side-by-side-summary), [`difference_table`](#difference-table), [`overlay_timeseries`](#overlay-timeseries), [`distribution_shift`](#distribution-shift)

_Last reviewed 2026-05-05 by macroforecast author._

### `overlay_timeseries`  --  operational

Time-series overlay of before / after for each series.

L2.5.A comparison output form ``overlay_timeseries``.

This option configures the ``comparison_output_form`` axis on the ``L2_5_A_comparison_axis`` sub-layer of L2.5; output is emitted under ``manifest.diagnostics/l2_5/L2_5_A_comparison_axis/`` alongside the other selected views.

**When to use**

Visual confirmation that cleaning preserves dynamics; spotting unexpected shape changes.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`side_by_side_summary`](#side-by-side-summary), [`difference_table`](#difference-table), [`distribution_shift`](#distribution-shift), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `side_by_side_summary`  --  operational

Side-by-side summary statistics for the chosen pair.

L2.5.A comparison output form ``side_by_side_summary``.

This option configures the ``comparison_output_form`` axis on the ``L2_5_A_comparison_axis`` sub-layer of L2.5; output is emitted under ``manifest.diagnostics/l2_5/L2_5_A_comparison_axis/`` alongside the other selected views.

**When to use**

Default tabular comparison; matches L1.5.B summary metrics applied to both pre and post.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`difference_table`](#difference-table), [`overlay_timeseries`](#overlay-timeseries), [`distribution_shift`](#distribution-shift), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._
