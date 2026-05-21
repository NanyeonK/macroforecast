# `quarterly_to_monthly_rule`

[Back to L2](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``quarterly_to_monthly_rule`` on sub-layer ``l2_a`` (layer ``l2``).

## Sub-layer

**l2_a**

## Axis metadata

- Default: `'step_backward'`
- Sweepable: True
- Status: operational

## Operational status summary

- Operational: 3 option(s)
- Future: 1 option(s)

## Options

### `linear_interpolation`  --  operational

Linear interpolation between quarterly observations.

Smoother than step_backward but introduces look-ahead unless used per-origin.

Configures the ``quarterly_to_monthly_rule`` axis on ``l2_a`` (layer ``l2``); the ``linear_interpolation`` value is materialised in the recipe's ``fixed_axes`` block under that sub-layer.

**When to use**

Studies with smooth quarterly series and per-origin alignment.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`step_backward`](#step-backward), [`chow_lin`](#chow-lin)

_Last reviewed 2026-05-04 by macroforecast author._

### `step_backward`  --  operational

Step-function: each month inherits the most-recent published quarterly value.

See [step_backward function page](../quarterly_to_monthly_rule/step_backward.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.freq_align_quarterly_to_monthly_clean``.

### `step_forward`  --  operational

Step-function: each month inherits the next-published quarterly value.

Use when later observations are informative for current state (rare in real-time work).

**When to use**

Hindsight-feasible studies (e.g., counterfactual nowcasts).

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`step_backward`](#step-backward), [`linear_interpolation`](#linear-interpolation)

_Last reviewed 2026-05-04 by macroforecast author._

### `chow_lin`  --  future

_(no schema description for `chow_lin`)_

> TBD: option doc not yet authored for this value. The encyclopedia falls back to the bare schema description above. PRs adding a full ``OptionDoc`` entry under ``macroforecast/scaffold/option_docs/l2.py`` are welcome.
