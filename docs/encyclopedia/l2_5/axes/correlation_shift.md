# `correlation_shift`

[Back to L2.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``correlation_shift`` on sub-layer ``L2_5_C_correlation_shift`` (layer ``l2_5``).

## Sub-layer

**L2_5_C_correlation_shift**

## Axis metadata

- Default: `'none'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 3 option(s)
- Future: 0 option(s)

## Options

### `delta_matrix`  --  operational

Heatmap of (post - pre) correlation matrix.

L2.5.C correlation-shift view ``delta_matrix``.

This option configures the ``correlation_shift`` axis on the ``L2_5_C_correlation_shift`` sub-layer of L2.5; output is emitted under ``manifest.diagnostics/l2_5/L2_5_C_correlation_shift/`` alongside the other selected views.

**When to use**

Detecting cleaning steps that distort dependence structure; large entries flag pairs whose joint behaviour was altered.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`pre_post_overlay`](#pre-post-overlay), [`none`](#none)

_Last reviewed 2026-05-05 by macroforecast author._

### `none`  --  operational

Skip correlation-shift diagnostic.

L2.5.C correlation-shift view ``none``.

This option configures the ``correlation_shift`` axis on the ``L2_5_C_correlation_shift`` sub-layer of L2.5; output is emitted under ``manifest.diagnostics/l2_5/L2_5_C_correlation_shift/`` alongside the other selected views.

**When to use**

When L2 only changes scale (no transform that reshapes correlations).

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`delta_matrix`](#delta-matrix), [`pre_post_overlay`](#pre-post-overlay)

_Last reviewed 2026-05-05 by macroforecast author._

### `pre_post_overlay`  --  operational

Side-by-side pre / post correlation heatmaps.

L2.5.C correlation-shift view ``pre_post_overlay``.

This option configures the ``correlation_shift`` axis on the ``L2_5_C_correlation_shift`` sub-layer of L2.5; output is emitted under ``manifest.diagnostics/l2_5/L2_5_C_correlation_shift/`` alongside the other selected views.

**When to use**

Default visual; reveals magnitude and sign of changes simultaneously.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`delta_matrix`](#delta-matrix), [`none`](#none)

_Last reviewed 2026-05-05 by macroforecast author._
