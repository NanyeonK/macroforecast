# `density_metrics`

[Back to L5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``density_metrics`` on sub-layer ``L5_A_metric_specification`` (layer ``l5``).

## Sub-layer

**L5_A_metric_specification**

## Axis metadata

- Default: `['log_score', 'crps']`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 4 option(s)
- Future: 0 option(s)

## Options

### `coverage_rate`  --  operational

Empirical coverage rate -- share of OOS observations falling within the nominal-α interval.

Density-forecast metric ``coverage_rate``. Should equal α (1 - α miscoverage) if the model is well-calibrated. Deviations indicate miscalibration: low coverage = intervals too narrow; high coverage = intervals too wide. Pair with ``interval_score`` to capture both calibration and sharpness.

**When to use**

Interval-calibration audits; reporting alongside interval_score.

**References**

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Gneiting & Raftery (2007) 'Strictly Proper Scoring Rules, Prediction, and Estimation', JASA 102(477): 359-378. (doi:10.1198/016214506000001437)
* Gneiting & Katzfuss (2014) 'Probabilistic Forecasting', Annual Review of Statistics and Its Application 1: 125-151.

**Related options**: [`log_score`](#log-score), [`crps`](#crps), [`interval_score`](#interval-score)

_Last reviewed 2026-05-05 by macroforecast author._

### `crps`  --  operational

Continuous ranked probability score -- generalisation of MAE to densities.

Density-forecast metric ``crps``. ``CRPS = ∫ (F̂(y) - 1{y ≥ y_obs})² dy``. Strictly-proper, expressed in the same units as the target. Reduces to MAE when the predictive distribution is a point mass at the predicted value. Standard density-score in weather / macro forecasting (Gneiting-Katzfuss 2014).

**When to use**

Distributional forecasts; comparing point and density forecasts on a common scale.

**References**

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Gneiting & Raftery (2007) 'Strictly Proper Scoring Rules, Prediction, and Estimation', JASA 102(477): 359-378. (doi:10.1198/016214506000001437)
* Gneiting & Katzfuss (2014) 'Probabilistic Forecasting', Annual Review of Statistics and Its Application 1: 125-151.

**Related options**: [`log_score`](#log-score), [`interval_score`](#interval-score), [`coverage_rate`](#coverage-rate)

_Last reviewed 2026-05-05 by macroforecast author._

### `interval_score`  --  operational

Winkler (1972) interval score -- jointly penalises miscoverage + interval width.

Density-forecast metric ``interval_score``. For a nominal-α interval ``[L, U]``: ``IS_α = (U - L) + (2/α)(L - y) 1{y < L} + (2/α)(y - U) 1{y > U}``. Lower = better. Strictly-proper for the α-level prediction interval; the natural metric when L4 emits ``forecast_object = interval``.

**When to use**

Prediction-interval evaluation; balancing tightness against coverage.

**References**

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Gneiting & Raftery (2007) 'Strictly Proper Scoring Rules, Prediction, and Estimation', JASA 102(477): 359-378. (doi:10.1198/016214506000001437)
* Gneiting & Katzfuss (2014) 'Probabilistic Forecasting', Annual Review of Statistics and Its Application 1: 125-151.
* Winkler (1972) 'A Decision-Theoretic Approach to Interval Estimation', JASA 67(337): 187-191.

**Related options**: [`log_score`](#log-score), [`crps`](#crps), [`coverage_rate`](#coverage-rate)

_Last reviewed 2026-05-05 by macroforecast author._

### `log_score`  --  operational

Logarithmic predictive density score -- ``log f̂(y_t)``.

Density-forecast metric ``log_score``. The strictly-proper scoring rule recommended by Gneiting & Raftery (2007). Equivalent to the Bayesian predictive log-likelihood. Larger = better. Requires ``forecast_object = density / quantile`` from L4.

When the predictive density is parametric (e.g. Gaussian) the score reduces to a closed-form involving the predictive mean / variance.

**When to use**

Default scoring rule for Bayesian forecasts; probabilistic horse-race ranking.

**References**

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Gneiting & Raftery (2007) 'Strictly Proper Scoring Rules, Prediction, and Estimation', JASA 102(477): 359-378. (doi:10.1198/016214506000001437)
* Gneiting & Katzfuss (2014) 'Probabilistic Forecasting', Annual Review of Statistics and Its Application 1: 125-151.

**Related options**: [`crps`](#crps), [`interval_score`](#interval-score), [`coverage_rate`](#coverage-rate)

_Last reviewed 2026-05-05 by macroforecast author._
