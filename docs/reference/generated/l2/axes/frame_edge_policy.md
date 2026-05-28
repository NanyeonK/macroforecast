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

See [truncate_to_balanced function page](../frame_edge_policy/truncate_to_balanced.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.truncate_to_balanced_clean``.

### `drop_unbalanced_series`  --  operational

Drop predictor columns that aren't observed across the full sample.

See [drop_unbalanced_series function page](../frame_edge_policy/drop_unbalanced_series.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.drop_unbalanced_series_clean``.

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

See [zero_fill_leading function page](../frame_edge_policy/zero_fill_leading.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.zero_fill_leading_clean``.
