# `lag_view`

[Back to L3.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``lag_view`` on sub-layer ``L3_5_D_lag_block_inspection`` (layer ``l3_5``).

## Sub-layer

**L3_5_D_lag_block_inspection**

## Axis metadata

- Default: `'multi'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 4 option(s)
- Future: 0 option(s)

## Options

### `autocorrelation_per_lag`  --  operational

ACF plot for each lag-block series.

L3.5.D lag view ``autocorrelation_per_lag``.

This option configures the ``lag_view`` axis on the ``L3_5_D_lag_block_inspection`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_D_lag_block_inspection/`` alongside the other selected views.

**When to use**

Standard time-series ACF view; informs choice of maximum lag length.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`lag_correlation_decay`](#lag-correlation-decay), [`partial_autocorrelation`](#partial-autocorrelation), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `lag_correlation_decay`  --  operational

Decay rate of lag autocorrelations.

L3.5.D lag view ``lag_correlation_decay``.

This option configures the ``lag_view`` axis on the ``L3_5_D_lag_block_inspection`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_D_lag_block_inspection/`` alongside the other selected views.

**When to use**

Choosing maximum lag length quantitatively -- the half-life of the ACF gives a natural cutoff.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`autocorrelation_per_lag`](#autocorrelation-per-lag), [`partial_autocorrelation`](#partial-autocorrelation), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `multi`  --  operational

Produce ACF + PACF + decay together.

L3.5.D lag view ``multi``.

This option configures the ``lag_view`` axis on the ``L3_5_D_lag_block_inspection`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_D_lag_block_inspection/`` alongside the other selected views.

**When to use**

Default rich lag audit. Activates the ``multi`` branch on L3.5.lag_view; combine with related options on the same sub-layer for a comprehensive diagnostic.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`autocorrelation_per_lag`](#autocorrelation-per-lag), [`lag_correlation_decay`](#lag-correlation-decay), [`partial_autocorrelation`](#partial-autocorrelation)

_Last reviewed 2026-05-05 by macroforecast author._

### `partial_autocorrelation`  --  operational

PACF plot for each lag-block series.

L3.5.D lag view ``partial_autocorrelation``.

This option configures the ``lag_view`` axis on the ``L3_5_D_lag_block_inspection`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_D_lag_block_inspection/`` alongside the other selected views.

**When to use**

Choosing AR(p) order; the lag at which PACF first hits the noise band suggests p.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`autocorrelation_per_lag`](#autocorrelation-per-lag), [`lag_correlation_decay`](#lag-correlation-decay), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._
