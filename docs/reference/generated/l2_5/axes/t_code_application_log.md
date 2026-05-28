# `t_code_application_log`

[Back to L2.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``t_code_application_log`` on sub-layer ``L2_5_D_cleaning_effect_summary`` (layer ``l2_5``).

## Sub-layer

**L2_5_D_cleaning_effect_summary**

## Axis metadata

- Default: `'summary'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 3 option(s)
- Future: 0 option(s)

## Options

### `none`  --  operational

Skip the tcode log.

L2.5.D tcode log option ``none``.

This option configures the ``t_code_application_log`` axis on the ``L2_5_D_cleaning_effect_summary`` sub-layer of L2.5; output is emitted under ``manifest.diagnostics/l2_5/L2_5_D_cleaning_effect_summary/`` alongside the other selected views.

**When to use**

When ``transform_policy = no_transform`` and no tcodes were applied.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'
* McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic Research', JBES 34(4): 574-589. (doi:10.1080/07350015.2015.1086655)

**Related options**: [`summary`](#summary), [`per_series_detail`](#per-series-detail)

_Last reviewed 2026-05-05 by macroforecast author._

### `per_series_detail`  --  operational

Per-series tcode applied + before/after means.

L2.5.D tcode log option ``per_series_detail``.

This option configures the ``t_code_application_log`` axis on the ``L2_5_D_cleaning_effect_summary`` sub-layer of L2.5; output is emitted under ``manifest.diagnostics/l2_5/L2_5_D_cleaning_effect_summary/`` alongside the other selected views.

**When to use**

Forensic audit of tcode application; useful when investigating unexpected post-tcode behaviour.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'
* McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic Research', JBES 34(4): 574-589. (doi:10.1080/07350015.2015.1086655)

**Related options**: [`summary`](#summary), [`none`](#none)

_Last reviewed 2026-05-05 by macroforecast author._

### `summary`  --  operational

Tcode usage histogram (counts per tcode).

L2.5.D tcode log option ``summary``.

This option configures the ``t_code_application_log`` axis on the ``L2_5_D_cleaning_effect_summary`` sub-layer of L2.5; output is emitted under ``manifest.diagnostics/l2_5/L2_5_D_cleaning_effect_summary/`` alongside the other selected views.

**When to use**

Default; quick cumulative summary of which tcodes were applied.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'
* McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic Research', JBES 34(4): 574-589. (doi:10.1080/07350015.2015.1086655)

**Related options**: [`per_series_detail`](#per-series-detail), [`none`](#none)

_Last reviewed 2026-05-05 by macroforecast author._
