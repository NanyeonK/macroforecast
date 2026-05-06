# `coverage_view`

[Back to L1.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``coverage_view`` on sub-layer ``L1_5_A_sample_coverage`` (layer ``l1_5``).

## Sub-layer

**L1_5_A_sample_coverage**

## Axis metadata

- Default: `'multi'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 4 option(s)
- Future: 0 option(s)

## Options

### `multi`  --  operational

Render every coverage view in a single composite output.

Composite view containing ``observation_count`` + ``per_series_start_end`` + ``panel_balance_matrix`` in one HTML / PDF report. Recommended default for exploratory data review.

**When to use**

First-pass exploratory data review covering all three coverage angles.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`observation_count`](#observation-count), [`per_series_start_end`](#per-series-start-end), [`panel_balance_matrix`](#panel-balance-matrix)

_Last reviewed 2026-05-05 by macroforecast author._

### `observation_count`  --  operational

Per-series observation count vs sample length.

Bar chart of ``n_obs`` per series over the active sample window. Highlights series that may be too short for the L4 estimator -- a Lasso fit needs roughly n_obs > 2 × n_predictors, and short series violate that constraint silently.

**When to use**

First-pass sanity check that no predictor is mostly missing.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`per_series_start_end`](#per-series-start-end), [`panel_balance_matrix`](#panel-balance-matrix), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `panel_balance_matrix`  --  operational

Binary observed/missing matrix over the full sample.

Heatmap with rows = series, columns = dates, cells = 1 (observed) or 0 (missing). Reveals structural breaks in coverage -- e.g. a block of series that all start in 1990, or a block that disappears after 2008. The ragged-edge pattern is best understood here before L1.E's sample-window rule trims the panel.

**When to use**

Visualising ragged-edge problems before applying L1.E sample-window rules.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`per_series_start_end`](#per-series-start-end)

_Last reviewed 2026-05-05 by macroforecast author._

### `per_series_start_end`  --  operational

First / last observation date per series.

Table of per-series ``(first_valid_date, last_valid_date)``. Catches stale series that stopped publishing (last date too old) or new series with too short a history (first date too recent). Critical before applying the L1.E sample-window rule.

**When to use**

Diagnosing balanced-vs-unbalanced panel decisions in L1.E.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`observation_count`](#observation-count), [`panel_balance_matrix`](#panel-balance-matrix)

_Last reviewed 2026-05-05 by macroforecast author._
