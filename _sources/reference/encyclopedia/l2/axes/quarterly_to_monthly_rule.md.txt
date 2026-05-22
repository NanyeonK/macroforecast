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

- Operational: 4 option(s)
- Future: 0 option(s)

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

### `chow_lin`  --  operational

Chow-Lin (1971) regression-based temporal disaggregation.

Implements the best linear unbiased interpolation procedure of Chow & Lin (1971). A monthly indicator series (``leaf_config.chow_lin_indicator``) is regressed on the quarterly observations using GLS with an AR(1) error structure; the fitted residuals are distributed proportionally across the three months of each quarter so the sum-constraint is preserved.

This is the canonical temporal disaggregation method in the macroeconomics literature. It is more accurate than step-function or linear-interpolation approaches when an informative monthly indicator is available. Requires ``leaf_config.chow_lin_indicator`` to name the monthly indicator column present in the panel.

**When to use**

Quarterly-to-monthly disaggregation when a monthly indicator series is available (e.g., IP as indicator for GDP).

**When NOT to use**

When no suitable monthly indicator exists -- use step_backward (conservative) or linear_interpolation (smooth) instead.

**References**

* Chow & Lin (1971) 'Best Linear Unbiased Interpolation, Distribution, and Extrapolation of Time Series by Related Series', Review of Economics and Statistics 53(4): 372-375. (doi:10.2307/1928739)

**Related options**: [`step_backward`](#step-backward), [`linear_interpolation`](#linear-interpolation)

_Last reviewed 2026-05-04 by macroforecast author._
