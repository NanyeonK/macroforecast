# `ensemble_view`

[Back to L4.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``ensemble_view`` on sub-layer ``L4_5_E_ensemble_diagnostics`` (layer ``l4_5``).

## Sub-layer

**L4_5_E_ensemble_diagnostics**

## Axis metadata

- Default: `'multi'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 4 option(s)
- Future: 0 option(s)

## Options

### `member_contribution`  --  operational

Per-member contribution to forecast variance.

L4.5.E ensemble view ``member_contribution``.

This option configures the ``ensemble_view`` axis on the ``L4_5_E_ensemble_diagnostics`` sub-layer of L4.5; output is emitted under ``manifest.diagnostics/l4_5/L4_5_E_ensemble_diagnostics/`` alongside the other selected views.

**When to use**

Identifying free-rider members that contribute little to the ensemble's predictive variance.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`weights_over_time`](#weights-over-time), [`weight_concentration`](#weight-concentration), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `multi`  --  operational

Render every ensemble diagnostic together.

L4.5.E ensemble view ``multi``.

This option configures the ``ensemble_view`` axis on the ``L4_5_E_ensemble_diagnostics`` sub-layer of L4.5; output is emitted under ``manifest.diagnostics/l4_5/L4_5_E_ensemble_diagnostics/`` alongside the other selected views.

**When to use**

Default rich ensemble audit. Activates the ``multi`` branch on L4.5.ensemble_view; combine with related options on the same sub-layer for a comprehensive diagnostic.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`weights_over_time`](#weights-over-time), [`weight_concentration`](#weight-concentration), [`member_contribution`](#member-contribution)

_Last reviewed 2026-05-05 by macroforecast author._

### `weight_concentration`  --  operational

Herfindahl / entropy of ensemble weights.

L4.5.E ensemble view ``weight_concentration``.

This option configures the ``ensemble_view`` axis on the ``L4_5_E_ensemble_diagnostics`` sub-layer of L4.5; output is emitted under ``manifest.diagnostics/l4_5/L4_5_E_ensemble_diagnostics/`` alongside the other selected views.

**When to use**

Quantifying ensemble diversity; concentrated weights = under-diversified ensemble.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`weights_over_time`](#weights-over-time), [`member_contribution`](#member-contribution), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `weights_over_time`  --  operational

Time-series of ensemble weights.

L4.5.E ensemble view ``weights_over_time``.

This option configures the ``ensemble_view`` axis on the ``L4_5_E_ensemble_diagnostics`` sub-layer of L4.5; output is emitted under ``manifest.diagnostics/l4_5/L4_5_E_ensemble_diagnostics/`` alongside the other selected views.

**When to use**

Tracking which member dominates over time; pairs with the L7 ``rolling_recompute`` for stability analysis.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`weight_concentration`](#weight-concentration), [`member_contribution`](#member-contribution), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._
