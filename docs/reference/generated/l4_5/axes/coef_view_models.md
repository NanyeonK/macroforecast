# `coef_view_models`

[Back to L4.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``coef_view_models`` on sub-layer ``L4_5_C_window_stability`` (layer ``l4_5``).

## Sub-layer

**L4_5_C_window_stability**

## Axis metadata

- Default: `'all_linear_models'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 2 option(s)
- Future: 0 option(s)

## Options

### `all_linear_models`  --  operational

Track coefficients across every linear model in the recipe.

L4.5.C coef-tracking model selector ``all_linear_models``.

This option configures the ``coef_view_models`` axis on the ``L4_5_C_window_stability`` sub-layer of L4.5; output is emitted under ``manifest.diagnostics/l4_5/L4_5_C_window_stability/`` alongside the other selected views.

**When to use**

Default broad audit; works automatically for ols / ridge / lasso / elastic_net.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`user_list`](#user-list)

_Last reviewed 2026-05-05 by macroforecast author._

### `user_list`  --  operational

Track coefficients only for a user-listed subset.

L4.5.C coef-tracking model selector ``user_list``.

This option configures the ``coef_view_models`` axis on the ``L4_5_C_window_stability`` sub-layer of L4.5; output is emitted under ``manifest.diagnostics/l4_5/L4_5_C_window_stability/`` alongside the other selected views.

**When to use**

Targeted audit when many linear models are active and only a few warrant inspection.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`all_linear_models`](#all-linear-models)

_Last reviewed 2026-05-05 by macroforecast author._
