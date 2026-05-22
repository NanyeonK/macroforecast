# `equal_predictive_test`

[Back to L6](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``equal_predictive_test`` on sub-layer ``L6_A_equal_predictive`` (layer ``l6``).

## Sub-layer

**L6_A_equal_predictive**

## Axis metadata

- Default: `'dm_diebold_mariano'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 5 option(s)
- Future: 0 option(s)

## Options

### `dm_diebold_mariano`  --  operational

Diebold-Mariano (1995) equal-predictive-ability test with Newey-West HAC SE.

See [dm_diebold_mariano function page](../equal_predictive_test/dm_diebold_mariano.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.dm_test``.

### `gw_giacomini_white`  --  operational

Giacomini-White (2006) conditional equal-predictive-ability test.

See [gw_giacomini_white function page](../equal_predictive_test/gw_giacomini_white.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.gw_test``.

### `dmp_multi_horizon`  --  operational

Diebold-Mariano-Pesaran joint multi-horizon test.

See [dmp_multi_horizon function page](../equal_predictive_test/dmp_multi_horizon.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.dmp_test``.

### `harvey_newbold_encompassing`  --  operational

Harvey-Leybourne-Newbold (1998) forecast-encompassing test.

See [harvey_newbold_encompassing function page](../equal_predictive_test/harvey_newbold_encompassing.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.hn_test``.

### `multi`  --  operational

Run DM + GW + DMP and stack the results.

Multi-test convenience option; emits a single output table with one row per test. Useful as a robustness check.

**When to use**

Comprehensive equal-predictive-ability audits.

**References**

* macroforecast design Part 3, L6: 'tests must report (statistic, p-value, kernel, lag) and respect HAC dependence-correction.'

**Related options**: [`dm_diebold_mariano`](#dm-diebold-mariano), [`gw_giacomini_white`](#gw-giacomini-white), [`dmp_multi_horizon`](#dmp-multi-horizon), [`harvey_newbold_encompassing`](#harvey-newbold-encompassing)

_Last reviewed 2026-05-05 by macroforecast author._
