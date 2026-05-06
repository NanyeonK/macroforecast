# `frame_edge_policy`

[Back to L2](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``frame_edge_policy`` on sub-layer ``l2_e`` (layer ``l2``).

## Sub-layer

**l2_e**

## Axis metadata

- Default: `'truncate_to_balanced'`
- Sweepable: True
- Status: operational

## Operational status summary

- Operational: 4 option(s)
- Future: 0 option(s)

## Options

### `truncate_to_balanced`  --  operational

Trim leading / trailing rows until every series is observed.

Makes the panel rectangular by removing rows where any predictor (or the target, depending on scope) is missing. Standard for factor-model-style studies that need a balanced panel.

**When to use**

Default for high-dimensional studies; pairs with em_factor imputation for the interior.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'
* Stock & Watson (2002) 'Macroeconomic Forecasting Using Diffusion Indexes', JBES 20(2).

**Related options**: [`drop_unbalanced_series`](#drop-unbalanced-series), [`keep_unbalanced`](#keep-unbalanced), [`zero_fill_leading`](#zero-fill-leading)

_Last reviewed 2026-05-04 by macroforecast author._

### `drop_unbalanced_series`  --  operational

Drop predictor columns that aren't observed across the full sample.

Trades predictor count for sample length. Useful when the recipe wants to keep early observations and is willing to lose late-arrival series.

**When to use**

Long-history studies (1959-) where late-introduction series should be excluded.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`truncate_to_balanced`](#truncate-to-balanced), [`keep_unbalanced`](#keep-unbalanced)

_Last reviewed 2026-05-04 by macroforecast author._

### `keep_unbalanced`  --  operational

Keep the panel's natural unbalanced shape.

Lets L4 estimators handle missingness directly. Required for some L4 families (LSTM/GRU/transformer) and for partial-data robustness studies.

**When to use**

Custom panels with intentional unbalanced structure; missing-data-robust models.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`truncate_to_balanced`](#truncate-to-balanced), [`drop_unbalanced_series`](#drop-unbalanced-series)

_Last reviewed 2026-05-04 by macroforecast author._

### `zero_fill_leading`  --  operational

Zero-fill leading missing predictor cells; preserve the rest.

Useful when leading NaN values block early-sample fits but interior NaN should remain visible to imputation.

**When to use**

Studies that want the early sample but accept zero-fill on leading edges.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`truncate_to_balanced`](#truncate-to-balanced), [`keep_unbalanced`](#keep-unbalanced)

_Last reviewed 2026-05-04 by macroforecast author._
