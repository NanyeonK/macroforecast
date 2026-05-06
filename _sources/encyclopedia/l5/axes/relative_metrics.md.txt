# `relative_metrics`

[Back to L5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``relative_metrics`` on sub-layer ``L5_A_metric_specification`` (layer ``l5``).

## Sub-layer

**L5_A_metric_specification**

## Axis metadata

- Default: `['relative_mse', 'r2_oos']`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 4 option(s)
- Future: 0 option(s)

## Options

### `mse_reduction`  --  operational

``1 - relative_mse`` -- positive means the candidate beats the benchmark.

Relative-loss metric ``mse_reduction``. Convenience reformulation that flips the sign so positive numbers indicate improvement. Common in macro-forecasting papers (e.g. Stock-Watson 2002 reports MSE reduction in %). Equivalent to ``1 - MSE_model / MSE_benchmark``.

**When to use**

Default reporting in horse-race tables when 'positive = better' is preferred.

**References**

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Campbell & Thompson (2008) 'Predicting Excess Stock Returns Out of Sample: Can Anything Beat the Historical Average?', Review of Financial Studies 21(4): 1509-1531. (doi:10.1093/rfs/hhm055)

**Related options**: [`relative_mse`](#relative-mse), [`relative_mae`](#relative-mae), [`r2_oos`](#r2-oos)

_Last reviewed 2026-05-05 by macroforecast author._

### `r2_oos`  --  operational

Out-of-sample R² (Campbell-Thompson 2008) -- ``1 - SSE_model / SSE_benchmark``.

Relative-loss metric ``r2_oos``. Standard return-predictability metric in finance (and increasingly in macro). Identical formula to ``mse_reduction`` when the benchmark is the historical mean. Campbell & Thompson (2008) popularised the metric for the empirical-asset-pricing literature.

**When to use**

Macro / financial forecasting tradition; literature-compatibility with CT-2008-era papers.

**References**

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Campbell & Thompson (2008) 'Predicting Excess Stock Returns Out of Sample: Can Anything Beat the Historical Average?', Review of Financial Studies 21(4): 1509-1531. (doi:10.1093/rfs/hhm055)

**Related options**: [`relative_mse`](#relative-mse), [`relative_mae`](#relative-mae), [`mse_reduction`](#mse-reduction)

_Last reviewed 2026-05-05 by macroforecast author._

### `relative_mae`  --  operational

Forecast MAE divided by the L4 ``is_benchmark`` model's MAE.

Relative-loss metric ``relative_mae``. L1-loss analogue of ``relative_mse``. Below 1 means the candidate beats the benchmark on absolute-loss criterion. Robust to heavy-tailed forecast errors.

**When to use**

Heavy-tailed targets where MSE is too sensitive to outliers.

**References**

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Diebold (2017) 'Forecasting in Economics, Business, Finance and Beyond', University of Pennsylvania (free online). <https://www.sas.upenn.edu/~fdiebold/Textbooks.html>

**Related options**: [`relative_mse`](#relative-mse), [`mse_reduction`](#mse-reduction), [`r2_oos`](#r2-oos)

_Last reviewed 2026-05-05 by macroforecast author._

### `relative_mse`  --  operational

Forecast MSE divided by the L4 ``is_benchmark`` model's MSE.

Relative-loss metric ``relative_mse``. ``MSE_model / MSE_benchmark``. The standard horse-race ratio. Below 1 means the candidate beats the benchmark; the L5.E ranking tables surface this column by default. Requires exactly one L4 model with ``is_benchmark = true`` (validator hard-rejects 0 or > 1 benchmarks).

**When to use**

Default reporting metric in horse-race tables; comparing candidate models against a fixed benchmark.

**References**

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Diebold (2017) 'Forecasting in Economics, Business, Finance and Beyond', University of Pennsylvania (free online). <https://www.sas.upenn.edu/~fdiebold/Textbooks.html>

**Related options**: [`relative_mae`](#relative-mae), [`mse_reduction`](#mse-reduction), [`r2_oos`](#r2-oos)

_Last reviewed 2026-05-05 by macroforecast author._
