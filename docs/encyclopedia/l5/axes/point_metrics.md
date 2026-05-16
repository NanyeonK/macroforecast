# `point_metrics`

[Back to L5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``point_metrics`` on sub-layer ``L5_A_metric_specification`` (layer ``l5``).

## Sub-layer

**L5_A_metric_specification**

## Axis metadata

- Default: `['mse', 'mae']`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 7 option(s)
- Future: 0 option(s)

## Options

### `mae`  --  operational

Mean absolute error -- ``(1/N) Σ |y_t - ŷ_t|``.

Point-forecast metric ``mae``. L1 loss; robust alternative to MSE. Equally weighs every absolute residual rather than penalising large errors super-linearly. The implicit decision rule under MAE is the median of the predictive distribution (vs the mean for MSE).

**When to use**

Heavy-tailed targets where extreme errors should not dominate; reporting in target units.

**When NOT to use**

When the squared-loss decision rule is what the user actually faces.

**References**

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Diebold (2017) 'Forecasting in Economics, Business, Finance and Beyond', University of Pennsylvania (free online). <https://www.sas.upenn.edu/~fdiebold/Textbooks.html>

**Related options**: [`mse`](#mse), [`rmse`](#rmse), [`medae`](#medae), [`mape`](#mape), [`theil_u1`](#theil-u1), [`theil_u2`](#theil-u2)

_Last reviewed 2026-05-05 by macroforecast author._

### `mape`  --  operational

Mean absolute percentage error -- ``(100/N) Σ |y_t - ŷ_t| / |y_t|``.

Point-forecast metric ``mape``. Scale-free percentage version of MAE. Allows comparing forecasts for targets on different scales (US GDP vs Korean GDP). Pathological when targets can be zero or near-zero -- the metric blows up. Hyndman & Koehler (2006) recommend MASE / sMAPE in those cases.

**When to use**

Cross-target / cross-country comparisons; reporting forecast accuracy in percentage terms.

**When NOT to use**

Targets that can be near zero (rates, growth rates) -- division by tiny ``|y_t|`` makes the metric explode.

**References**

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Diebold (2017) 'Forecasting in Economics, Business, Finance and Beyond', University of Pennsylvania (free online). <https://www.sas.upenn.edu/~fdiebold/Textbooks.html>
* Hyndman & Koehler (2006) 'Another look at measures of forecast accuracy', International Journal of Forecasting 22(4): 679-688. (doi:10.1016/j.ijforecast.2006.03.001)

**Related options**: [`mse`](#mse), [`rmse`](#rmse), [`mae`](#mae), [`medae`](#medae), [`theil_u1`](#theil-u1), [`theil_u2`](#theil-u2)

_Last reviewed 2026-05-05 by macroforecast author._

### `medae`  --  operational

Median absolute error -- ``median |y_t - ŷ_t|``.

Point-forecast metric ``medae``. Maximally robust point-forecast metric: substitution by median completely insulates the score from a constant-share of extreme residuals. Common in robust-statistics papers; rarer in mainstream forecasting.

**When to use**

Pathologically heavy-tailed errors (financial crises, regime shifts).

**When NOT to use**

Standard reporting -- mean-based metrics are the convention.

**References**

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Diebold (2017) 'Forecasting in Economics, Business, Finance and Beyond', University of Pennsylvania (free online). <https://www.sas.upenn.edu/~fdiebold/Textbooks.html>

**Related options**: [`mse`](#mse), [`rmse`](#rmse), [`mae`](#mae), [`mape`](#mape), [`theil_u1`](#theil-u1), [`theil_u2`](#theil-u2)

_Last reviewed 2026-05-05 by macroforecast author._

### `mse`  --  operational

Mean squared error -- ``(1/N) Σ (y_t - ŷ_t)²``.

Point-forecast metric ``mse``. The classical quadratic-loss metric. Optimal under Gaussian-residual / squared-loss decision theory; the L4 fit objective for OLS / ridge / elastic net is its in-sample version. MSE penalises large residuals super-linearly, so a single outlier in the OOS sample can dominate the score.

**When to use**

Default for Gaussian-residual problems; horse-race ranking under squared-loss decision rules.

**When NOT to use**

Heavy-tailed forecast errors -- a single outlier dominates the score; consider MAE or MedAE instead.

**References**

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Diebold (2017) 'Forecasting in Economics, Business, Finance and Beyond', University of Pennsylvania (free online). <https://www.sas.upenn.edu/~fdiebold/Textbooks.html>

**Related options**: [`rmse`](#rmse), [`mae`](#mae), [`medae`](#medae), [`mape`](#mape), [`theil_u1`](#theil-u1), [`theil_u2`](#theil-u2)

_Last reviewed 2026-05-05 by macroforecast author._

### `rmse`  --  operational

Root mean squared error -- ``√MSE``.

Point-forecast metric ``rmse``. Same ranking as MSE but expressed in target units (rather than squared target units). Standard reporting metric in macro / finance papers; pairs naturally with confidence-band charts since RMSE has the same units as the prediction interval.

**When to use**

Reporting forecast accuracy in target units.

**When NOT to use**

Heavy-tailed errors -- inherits MSE's outlier sensitivity.

**References**

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Diebold (2017) 'Forecasting in Economics, Business, Finance and Beyond', University of Pennsylvania (free online). <https://www.sas.upenn.edu/~fdiebold/Textbooks.html>

**Related options**: [`mse`](#mse), [`mae`](#mae), [`medae`](#medae), [`mape`](#mape), [`theil_u1`](#theil-u1), [`theil_u2`](#theil-u2)

_Last reviewed 2026-05-05 by macroforecast author._

### `theil_u1`  --  operational

Theil's U1 inequality coefficient -- bounded in ``[0, 1]``.

See [theil_u1 function page](../point_metrics/theil_u1.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.theil_u1``.

### `theil_u2`  --  operational

Theil's U2 inequality coefficient -- ratio of forecast MSE to no-change MSE.

Point-forecast metric ``theil_u2``. ``U₂ = √(Σ (ŷ_t - y_t)² / Σ (y_{t-1} - y_t)²)``. ``U₂ < 1`` means the forecast beats the random-walk benchmark. Standard sanity-check ratio in macro forecasting -- if ``U₂ ≥ 1`` the model is no better than 'tomorrow looks like today'.

**When to use**

Sanity-checking against the random-walk benchmark; macro-forecasting tradition.

**When NOT to use**

When a custom benchmark (not random walk) is preferred -- use ``relative_mse`` instead.

**References**

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Theil (1966) 'Applied Economic Forecasting', North-Holland (Chapter 2: Inequality coefficients).

**Related options**: [`mse`](#mse), [`rmse`](#rmse), [`mae`](#mae), [`medae`](#medae), [`mape`](#mape), [`theil_u1`](#theil-u1)

_Last reviewed 2026-05-05 by macroforecast author._
