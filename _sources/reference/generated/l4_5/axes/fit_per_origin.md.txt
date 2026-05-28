# `fit_per_origin`

[Back to L4.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``fit_per_origin`` on sub-layer ``L4_5_A_in_sample_fit`` (layer ``l4_5``).

## Sub-layer

**L4_5_A_in_sample_fit**

## Axis metadata

- Default: `'last_origin_only'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 3 option(s)
- Future: 0 option(s)

## Options

### `all_origins`  --  operational

Compute fit views for every OOS origin.

L4.5.A fit-per-origin cadence ``all_origins``.

This option configures the ``fit_per_origin`` axis on the ``L4_5_A_in_sample_fit`` sub-layer of L4.5; output is emitted under ``manifest.diagnostics/l4_5/L4_5_A_in_sample_fit/`` alongside the other selected views.

**When to use**

Detailed walk-forward audit; expensive but thorough.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`every_n_origins`](#every-n-origins), [`last_origin_only`](#last-origin-only)

_Last reviewed 2026-05-05 by macroforecast author._

### `every_n_origins`  --  operational

Compute fit views every n origins.

L4.5.A fit-per-origin cadence ``every_n_origins``.

This option configures the ``fit_per_origin`` axis on the ``L4_5_A_in_sample_fit`` sub-layer of L4.5; output is emitted under ``manifest.diagnostics/l4_5/L4_5_A_in_sample_fit/`` alongside the other selected views.

**When to use**

Compromise between coverage and runtime; ``params.every_n`` controls cadence.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`all_origins`](#all-origins), [`last_origin_only`](#last-origin-only)

_Last reviewed 2026-05-05 by macroforecast author._

### `last_origin_only`  --  operational

Compute fit views only for the last training window.

L4.5.A fit-per-origin cadence ``last_origin_only``.

This option configures the ``fit_per_origin`` axis on the ``L4_5_A_in_sample_fit`` sub-layer of L4.5; output is emitted under ``manifest.diagnostics/l4_5/L4_5_A_in_sample_fit/`` alongside the other selected views.

**When to use**

Default; cheapest summary while still capturing the most-informative model state.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`all_origins`](#all-origins), [`every_n_origins`](#every-n-origins)

_Last reviewed 2026-05-05 by macroforecast author._
