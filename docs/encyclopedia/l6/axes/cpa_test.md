# `cpa_test`

[Back to L6](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``cpa_test`` on sub-layer ``L6_C_cpa`` (layer ``l6``).

## Sub-layer

**L6_C_cpa**

## Axis metadata

- Default: `'giacomini_rossi_2010'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 3 option(s)
- Future: 0 option(s)

## Options

### `giacomini_rossi_2010`  --  operational

Giacomini-Rossi (2010) rolling-window fluctuation test.

Rolling-window analogue of the GW test that tracks the evolution of predictive ability over time. v0.25 ships the simulated-CV table for ``(m/T, alpha)`` pairs used to compute exact critical values.

**When to use**

Detecting whether predictive ability is stable across the OOS sample.

**References**

* macroforecast design Part 3, L6: 'tests must report (statistic, p-value, kernel, lag) and respect HAC dependence-correction.'
* Giacomini & Rossi (2010) 'Forecast Comparisons in Unstable Environments', JAE 25(4): 595-620.

**Related options**: [`rossi_sekhposyan`](#rossi-sekhposyan), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `rossi_sekhposyan`  --  operational

Rossi-Sekhposyan (2011/2016) one-time / instabilities tests.

Companion suite of conditional predictive ability tests based on monitoring statistics over the OOS sample. Detects structural breaks in relative forecast performance.

**When to use**

Detecting one-off regime shifts in predictive ability.

**References**

* macroforecast design Part 3, L6: 'tests must report (statistic, p-value, kernel, lag) and respect HAC dependence-correction.'
* Rossi & Sekhposyan (2016) 'Forecast Rationality Tests in the Presence of Instabilities', JAE 31(3): 507-532.

**Related options**: [`giacomini_rossi_2010`](#giacomini-rossi-2010)

_Last reviewed 2026-05-05 by macroforecast author._

### `multi`  --  operational

Run all CPA tests and stack the results.

Multi-test convenience option; emits one row per CPA test.

Configures the ``cpa_test`` axis on ``L6_C_cpa`` (layer ``l6``); the ``multi`` value is materialised in the recipe's ``fixed_axes`` block under that sub-layer.

**When to use**

Comprehensive CPA audits. Selecting ``multi`` on ``l6.cpa_test`` activates this branch of the layer's runtime.

**References**

* macroforecast design Part 3, L6: 'tests must report (statistic, p-value, kernel, lag) and respect HAC dependence-correction.'

**Related options**: [`giacomini_rossi_2010`](#giacomini-rossi-2010), [`rossi_sekhposyan`](#rossi-sekhposyan)

_Last reviewed 2026-05-05 by macroforecast author._
