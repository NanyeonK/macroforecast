# `primary_metric`

[Back to L5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``primary_metric`` on sub-layer ``L5_A_metric_specification`` (layer ``l5``).

## Sub-layer

**L5_A_metric_specification**

## Axis metadata

- Default: `'mse'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 13 option(s)
- Future: 0 option(s)

## Options

### `crps`  --  operational

Continuous ranked probability score -- generalisation of MAE to densities.

Primary metric used for the L5.A summary table and the L5.E ranking. ``CRPS = ∫ (F̂(y) - 1{y ≥ y_obs})² dy``. Strictly-proper, expressed in the same units as the target. Reduces to MAE when the predictive distribution is a point mass at the predicted value. Standard density-score in weather / macro forecasting (Gneiting-Katzfuss 2014).

**When to use**

Distributional forecasts; comparing point and density forecasts on a common scale.

**References**

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Gneiting & Raftery (2007) 'Strictly Proper Scoring Rules, Prediction, and Estimation', JASA 102(477): 359-378. (doi:10.1198/016214506000001437)
* Gneiting & Katzfuss (2014) 'Probabilistic Forecasting', Annual Review of Statistics and Its Application 1: 125-151.

**Related options**: [`mse`](#mse), [`rmse`](#rmse), [`mae`](#mae), [`medae`](#medae)

_Last reviewed 2026-05-05 by macroforecast author._

### `log_score`  --  operational

Logarithmic predictive density score -- ``log f̂(y_t)``.

Primary metric used for the L5.A summary table and the L5.E ranking. The strictly-proper scoring rule recommended by Gneiting & Raftery (2007). Equivalent to the Bayesian predictive log-likelihood. Larger = better. Requires ``forecast_object = density / quantile`` from L4.

When the predictive density is parametric (e.g. Gaussian) the score reduces to a closed-form involving the predictive mean / variance.

**When to use**

Default scoring rule for Bayesian forecasts; probabilistic horse-race ranking.

**References**

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Gneiting & Raftery (2007) 'Strictly Proper Scoring Rules, Prediction, and Estimation', JASA 102(477): 359-378. (doi:10.1198/016214506000001437)

**Related options**: [`mse`](#mse), [`rmse`](#rmse), [`mae`](#mae), [`medae`](#medae)

_Last reviewed 2026-05-05 by macroforecast author._

### `mae`  --  operational

Mean absolute error -- ``(1/N) Σ |y_t - ŷ_t|``.

Primary metric used for the L5.A summary table and the L5.E ranking. L1 loss; robust alternative to MSE. Equally weighs every absolute residual rather than penalising large errors super-linearly. The implicit decision rule under MAE is the median of the predictive distribution (vs the mean for MSE).

**When to use**

Heavy-tailed targets where extreme errors should not dominate; reporting in target units.

**When NOT to use**

When the squared-loss decision rule is what the user actually faces.

**References**

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Diebold (2017) 'Forecasting in Economics, Business, Finance and Beyond', University of Pennsylvania (free online). <https://www.sas.upenn.edu/~fdiebold/Textbooks.html>

**Related options**: [`mse`](#mse), [`rmse`](#rmse), [`medae`](#medae), [`mape`](#mape)

_Last reviewed 2026-05-05 by macroforecast author._

### `mape`  --  operational

Mean absolute percentage error -- ``(100/N) Σ |y_t - ŷ_t| / |y_t|``.

Primary metric used for the L5.A summary table and the L5.E ranking. Scale-free percentage version of MAE. Allows comparing forecasts for targets on different scales (US GDP vs Korean GDP). Pathological when targets can be zero or near-zero -- the metric blows up. Hyndman & Koehler (2006) recommend MASE / sMAPE in those cases.

**When to use**

Cross-target / cross-country comparisons; reporting forecast accuracy in percentage terms.

**When NOT to use**

Targets that can be near zero (rates, growth rates) -- division by tiny ``|y_t|`` makes the metric explode.

**References**

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Hyndman & Koehler (2006) 'Another look at measures of forecast accuracy', International Journal of Forecasting 22(4): 679-688. (doi:10.1016/j.ijforecast.2006.03.001)

**Related options**: [`mse`](#mse), [`rmse`](#rmse), [`mae`](#mae), [`medae`](#medae)

_Last reviewed 2026-05-05 by macroforecast author._

### `medae`  --  operational

Median absolute error -- ``median |y_t - ŷ_t|``.

Primary metric used for the L5.A summary table and the L5.E ranking. Maximally robust point-forecast metric: substitution by median completely insulates the score from a constant-share of extreme residuals. Common in robust-statistics papers; rarer in mainstream forecasting.

**When to use**

Pathologically heavy-tailed errors (financial crises, regime shifts).

**When NOT to use**

Standard reporting -- mean-based metrics are the convention.

**References**

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'

**Related options**: [`mse`](#mse), [`rmse`](#rmse), [`mae`](#mae), [`mape`](#mape)

_Last reviewed 2026-05-05 by macroforecast author._

### `mse`  --  operational

Mean squared error -- ``(1/N) Σ (y_t - ŷ_t)²``.

Primary metric used for the L5.A summary table and the L5.E ranking. The classical quadratic-loss metric. Optimal under Gaussian-residual / squared-loss decision theory; the L4 fit objective for OLS / ridge / elastic net is its in-sample version. MSE penalises large residuals super-linearly, so a single outlier in the OOS sample can dominate the score.

**When to use**

Default for Gaussian-residual problems; horse-race ranking under squared-loss decision rules.

**When NOT to use**

Heavy-tailed forecast errors -- a single outlier dominates the score; consider MAE or MedAE instead.

**References**

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Diebold (2017) 'Forecasting in Economics, Business, Finance and Beyond', University of Pennsylvania (free online). <https://www.sas.upenn.edu/~fdiebold/Textbooks.html>

**Related options**: [`rmse`](#rmse), [`mae`](#mae), [`medae`](#medae), [`mape`](#mape)

_Last reviewed 2026-05-05 by macroforecast author._

### `mse_reduction`  --  operational

``1 - relative_mse`` -- positive means the candidate beats the benchmark.

Primary metric used for the L5.A summary table and the L5.E ranking. Convenience reformulation that flips the sign so positive numbers indicate improvement. Common in macro-forecasting papers (e.g. Stock-Watson 2002 reports MSE reduction in %). Equivalent to ``1 - MSE_model / MSE_benchmark``.

**When to use**

Default reporting in horse-race tables when 'positive = better' is preferred.

**References**

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Campbell & Thompson (2008) 'Predicting Excess Stock Returns Out of Sample: Can Anything Beat the Historical Average?', Review of Financial Studies 21(4): 1509-1531. (doi:10.1093/rfs/hhm055)

**Related options**: [`mse`](#mse), [`rmse`](#rmse), [`mae`](#mae), [`medae`](#medae)

_Last reviewed 2026-05-05 by macroforecast author._

### `r2_oos`  --  operational

Out-of-sample R² (Campbell-Thompson 2008) -- ``1 - SSE_model / SSE_benchmark``.

Primary metric used for the L5.A summary table and the L5.E ranking. Standard return-predictability metric in finance (and increasingly in macro). Identical formula to ``mse_reduction`` when the benchmark is the historical mean. Campbell & Thompson (2008) popularised the metric for the empirical-asset-pricing literature.

**When to use**

Macro / financial forecasting tradition; literature-compatibility with CT-2008-era papers.

**References**

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Campbell & Thompson (2008) 'Predicting Excess Stock Returns Out of Sample: Can Anything Beat the Historical Average?', Review of Financial Studies 21(4): 1509-1531. (doi:10.1093/rfs/hhm055)

**Related options**: [`mse`](#mse), [`rmse`](#rmse), [`mae`](#mae), [`medae`](#medae)

_Last reviewed 2026-05-05 by macroforecast author._

### `relative_mae`  --  operational

Forecast MAE divided by the L4 ``is_benchmark`` model's MAE.

Primary metric used for the L5.A summary table and the L5.E ranking. L1-loss analogue of ``relative_mse``. Below 1 means the candidate beats the benchmark on absolute-loss criterion. Robust to heavy-tailed forecast errors.

**When to use**

Heavy-tailed targets where MSE is too sensitive to outliers.

**References**

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'

**Related options**: [`mse`](#mse), [`rmse`](#rmse), [`mae`](#mae), [`medae`](#medae)

_Last reviewed 2026-05-05 by macroforecast author._

### `relative_mse`  --  operational

Forecast MSE divided by the L4 ``is_benchmark`` model's MSE.

Primary metric used for the L5.A summary table and the L5.E ranking. ``MSE_model / MSE_benchmark``. The standard horse-race ratio. Below 1 means the candidate beats the benchmark; the L5.E ranking tables surface this column by default. Requires exactly one L4 model with ``is_benchmark = true`` (validator hard-rejects 0 or > 1 benchmarks).

**When to use**

Default reporting metric in horse-race tables; comparing candidate models against a fixed benchmark.

**References**

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Diebold (2017) 'Forecasting in Economics, Business, Finance and Beyond', University of Pennsylvania (free online). <https://www.sas.upenn.edu/~fdiebold/Textbooks.html>

**Related options**: [`mse`](#mse), [`rmse`](#rmse), [`mae`](#mae), [`medae`](#medae)

_Last reviewed 2026-05-05 by macroforecast author._

### `rmse`  --  operational

Root mean squared error -- ``√MSE``.

Primary metric used for the L5.A summary table and the L5.E ranking. Same ranking as MSE but expressed in target units (rather than squared target units). Standard reporting metric in macro / finance papers; pairs naturally with confidence-band charts since RMSE has the same units as the prediction interval.

**When to use**

Reporting forecast accuracy in target units.

**When NOT to use**

Heavy-tailed errors -- inherits MSE's outlier sensitivity.

**References**

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Diebold (2017) 'Forecasting in Economics, Business, Finance and Beyond', University of Pennsylvania (free online). <https://www.sas.upenn.edu/~fdiebold/Textbooks.html>

**Related options**: [`mse`](#mse), [`mae`](#mae), [`medae`](#medae), [`mape`](#mape)

_Last reviewed 2026-05-05 by macroforecast author._

### `theil_u1`  --  operational

Theil's U1 inequality coefficient -- bounded in ``[0, 1]``.

Primary metric used for the L5.A summary table and the L5.E ranking. ``U₁ = √MSE / (√(1/N Σ y²) + √(1/N Σ ŷ²))``. Bounded between 0 (perfect forecast) and 1 (worst possible). Theil's original 1966 metric; less commonly used today than U2 because the denominator's interpretation is less intuitive.

**When to use**

Long-run macro forecasting tradition; comparability with Theil-1966-era papers.

**When NOT to use**

Modern reporting -- U2 is more interpretable as a ratio against the no-change benchmark.

**References**

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Theil (1966) 'Applied Economic Forecasting', North-Holland (Chapter 2: Inequality coefficients).

**Related options**: [`mse`](#mse), [`rmse`](#rmse), [`mae`](#mae), [`medae`](#medae)

_Last reviewed 2026-05-05 by macroforecast author._

### `theil_u2`  --  operational

Theil's U2 inequality coefficient -- ratio of forecast MSE to no-change MSE.

Primary metric used for the L5.A summary table and the L5.E ranking. ``U₂ = √(Σ (ŷ_t - y_t)² / Σ (y_{t-1} - y_t)²)``. ``U₂ < 1`` means the forecast beats the random-walk benchmark. Standard sanity-check ratio in macro forecasting -- if ``U₂ ≥ 1`` the model is no better than 'tomorrow looks like today'.

**When to use**

Sanity-checking against the random-walk benchmark; macro-forecasting tradition.

**When NOT to use**

When a custom benchmark (not random walk) is preferred -- use ``relative_mse`` instead.

**References**

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Theil (1966) 'Applied Economic Forecasting', North-Holland (Chapter 2: Inequality coefficients).

**Related options**: [`mse`](#mse), [`rmse`](#rmse), [`mae`](#mae), [`medae`](#medae)

_Last reviewed 2026-05-05 by macroforecast author._
