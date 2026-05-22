# `nested_test`

[Back to L6](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``nested_test`` on sub-layer ``L6_B_nested`` (layer ``l6``).

## Sub-layer

**L6_B_nested**

## Axis metadata

- Default: `'clark_west'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 4 option(s)
- Future: 0 option(s)

## Options

### `clark_west`  --  operational

Clark-West (2007) MSE-adjusted nested-model predictive ability test.

See [clark_west function page](../nested_test/clark_west.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.cw_test``.

### `enc_new`  --  operational

Enc-New forecast encompassing test (Clark-McCracken 2001).

See [enc_new function page](../nested_test/enc_new.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.enc_new_test``.

### `enc_t`  --  operational

Enc-T forecast encompassing test (Ericsson 1992 t-form).

See [enc_t function page](../nested_test/enc_t.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.enc_t_test``.

### `multi`  --  operational

Run clark_west + enc_new + enc_t and stack the results.

Multi-test convenience option; emits one row per nested test.

Configures the ``nested_test`` axis on ``L6_B_nested`` (layer ``l6``); the ``multi`` value is materialised in the recipe's ``fixed_axes`` block under that sub-layer.

**When to use**

Comprehensive nested-model evaluation audits.

**References**

* macroforecast design Part 3, L6: 'tests must report (statistic, p-value, kernel, lag) and respect HAC dependence-correction.'

**Related options**: [`clark_west`](#clark-west), [`enc_new`](#enc-new), [`enc_t`](#enc-t)

_Last reviewed 2026-05-05 by macroforecast author._
