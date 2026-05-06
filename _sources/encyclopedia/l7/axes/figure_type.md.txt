# `figure_type`

[Back to L7](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``figure_type`` on sub-layer ``L7_B_output_shape_export`` (layer ``l7``).

## Sub-layer

**L7_B_output_shape_export**

## Axis metadata

- Default: `'auto'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 6 option(s)
- Future: 0 option(s)

## Options

### `auto`  --  operational

Pick the figure type matching the importance op's default mapping.

Each L7.A op declares its canonical figure type (``shap_*`` → bar/beeswarm; ``partial_dependence`` → lineplot; ``shap_interaction`` → heatmap; ``rolling_recompute`` → heatmap; etc.). Setting ``figure_type = auto`` honours that default.

**When to use**

Default; lets each L7.A op choose the canonical figure for its output.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'

**Related options**: [`bar`](#bar), [`boxplot`](#boxplot), [`heatmap`](#heatmap), [`lineplot`](#lineplot), [`scatter`](#scatter)

_Last reviewed 2026-05-05 by macroforecast author._

### `bar`  --  operational

Horizontal bar chart -- one bar per feature, length = importance score.

The standard global-importance visualisation. Renders features sorted by mean-``|importance|`` so the most important variables surface at the top of the chart. Pair with ``output_table_format = wide`` for direct table-figure cross-reference.

**When to use**

Global importance rankings (linear coefficients, permutation, mean SHAP).

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'

**Related options**: [`auto`](#auto), [`boxplot`](#boxplot), [`heatmap`](#heatmap), [`lineplot`](#lineplot), [`scatter`](#scatter)

_Last reviewed 2026-05-05 by macroforecast author._

### `boxplot`  --  operational

Boxplot of per-fold / per-bootstrap importance distributions.

Renders each feature as a box capturing the distribution of its importance score across folds (cross-validation, bootstrap, rolling windows). Reveals stability information that a single bar cannot convey.

**When to use**

Stability-of-importance audits; ``bootstrap_jackknife`` / ``rolling_recompute`` outputs.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'

**Related options**: [`auto`](#auto), [`bar`](#bar), [`heatmap`](#heatmap), [`lineplot`](#lineplot), [`scatter`](#scatter)

_Last reviewed 2026-05-05 by macroforecast author._

### `heatmap`  --  operational

Two-axis heatmap (feature × time / model / state).

Visualises importance across an additional dimension. Used for time-varying importance (``rolling_recompute``), per-state aggregation (``group_aggregate`` over FRED-SD blocks), and pairwise interaction strength (``shap_interaction``). The ``us_state_choropleth`` figure is a specialised heatmap on the US state grid.

**When to use**

Time-varying importance, FRED-SD state choropleth, group-aggregate matrices.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'

**Related options**: [`auto`](#auto), [`bar`](#bar), [`boxplot`](#boxplot), [`lineplot`](#lineplot), [`scatter`](#scatter)

_Last reviewed 2026-05-05 by macroforecast author._

### `lineplot`  --  operational

Line plot of importance over time / origin.

Tracks importance evolution across walk-forward origins. Pair with ``rolling_recompute`` to surface trends in which features matter as new data arrives.

**When to use**

Tracking importance evolution across walk-forward origins.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'

**Related options**: [`auto`](#auto), [`bar`](#bar), [`boxplot`](#boxplot), [`heatmap`](#heatmap), [`scatter`](#scatter)

_Last reviewed 2026-05-05 by macroforecast author._

### `scatter`  --  operational

Scatter plot (e.g. SHAP value vs feature value).

PDP / ALE / SHAP dependence-plot family. Each point is a single observation; the x-axis is the feature value, the y-axis is the importance contribution. Reveals non-linearity in the model's response.

**When to use**

PDP / ALE / SHAP dependence-plot family.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'

**Related options**: [`auto`](#auto), [`bar`](#bar), [`boxplot`](#boxplot), [`heatmap`](#heatmap), [`lineplot`](#lineplot)

_Last reviewed 2026-05-05 by macroforecast author._
