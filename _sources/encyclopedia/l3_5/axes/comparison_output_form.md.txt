# `comparison_output_form`

[Back to L3.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``comparison_output_form`` on sub-layer ``L3_5_A_comparison_axis`` (layer ``l3_5``).

## Sub-layer

**L3_5_A_comparison_axis**

## Axis metadata

- Default: `'multi'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 4 option(s)
- Future: 0 option(s)

## Options

### `dimension_summary`  --  operational

Compare panel shape (N, T, NaN%) across stages.

L3.5.A comparison output form ``dimension_summary``.

This option configures the ``comparison_output_form`` axis on the ``L3_5_A_comparison_axis`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_A_comparison_axis/`` alongside the other selected views.

**When to use**

Verifying expected dimensionality changes -- e.g. confirming PCA reduced 100 columns to 5 factors.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`side_by_side`](#side-by-side), [`distribution_shift`](#distribution-shift), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `distribution_shift`  --  operational

KS / histogram comparison across stages.

L3.5.A comparison output form ``distribution_shift``.

This option configures the ``comparison_output_form`` axis on the ``L3_5_A_comparison_axis`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_A_comparison_axis/`` alongside the other selected views.

**When to use**

Detecting feature transforms that materially reshape distributions (e.g. wavelet / fourier expansions).

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`side_by_side`](#side-by-side), [`dimension_summary`](#dimension-summary), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `multi`  --  operational

Render every comparison output together.

L3.5.A comparison output form ``multi``.

This option configures the ``comparison_output_form`` axis on the ``L3_5_A_comparison_axis`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_A_comparison_axis/`` alongside the other selected views.

**When to use**

Comprehensive feature-stage audit; recommended for first-time runs.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`side_by_side`](#side-by-side), [`dimension_summary`](#dimension-summary), [`distribution_shift`](#distribution-shift)

_Last reviewed 2026-05-05 by macroforecast author._

### `side_by_side`  --  operational

Stage-by-stage side-by-side panel summaries.

L3.5.A comparison output form ``side_by_side``.

This option configures the ``comparison_output_form`` axis on the ``L3_5_A_comparison_axis`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_A_comparison_axis/`` alongside the other selected views.

**When to use**

Default multi-stage view; matches L2.5.A output style for consistency.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`dimension_summary`](#dimension-summary), [`distribution_shift`](#distribution-shift), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._
