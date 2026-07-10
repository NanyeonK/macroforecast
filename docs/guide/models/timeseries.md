# Classical time series

[Back to Models and Features](../model_overview.md)

Classical time series models work from the target's own history, including autoregressions, ARIMA, and exponential smoothing, and serve as the standard benchmarks.

Pass any model string below as `Arm(model=...)`. Extra names an optional dependency, Scaling flags whether predictors should be standardized, and Tunable counts the hyperparameters the search space exposes.

| Model string | Description | Input | Extra | Scaling | Recommended preprocessing | Tunable |
| --- | --- | --- | --- | --- | --- | --- |
| `ar` | Univariate autoregression. | supervised | none | no | default | 1 |
| `ar_bic` | Target-only AR with internal information-criterion lag selection. | target | none | no | default | 2 |
| `arima` | (Seasonal) ARIMA model. | target | none | no | default | 1 |
| `auto_arima` | Automatic (seasonal) ARIMA order selection (forecast::auto.arima). | target | none | no | default | 0 |
| `bvar_minnesota` | FAVAR::BVAR / bvartools Minnesota-prior Bayesian VAR posterior sampler. | panel | none | no | default | 3 |
| `bvar_normal_inverse_wishart` | FAVAR::BVAR-aligned Bayesian VAR with normal/inverse-Wishart prior controls. | panel | none | no | default | 1 |
| `ets` | Statsmodels ETS target-only forecasting model. | target | none | no | default | 0 |
| `hist_mean` | Historical (prevailing) mean benchmark of the transformed target. | target | none | no | default | 0 |
| `holt_winters` | Holt-Winters exponential smoothing target-only forecasting model. | target | none | no | default | 0 |
| `naive` | Random-walk (naive) baseline: carry the last value forward (forecast::naive). | target | none | no | default | 0 |
| `random_walk_drift` | Random-walk-with-drift baseline (forecast::rwf(drift=TRUE)). | target | none | no | default | 0 |
| `seasonal_naive` | Seasonal-naive baseline: repeat the last seasonal cycle (forecast::snaive). | target | none | no | default | 0 |
| `stlf` | STL decomposition + forecast of the seasonally-adjusted series (forecast::stlf). | target | none | no | default | 0 |
| `theta_method` | Theta method target-only forecasting model. | target | none | no | default | 0 |
| `var` | R vars::VAR-aligned vector autoregression point forecast. | panel | none | no | default | 1 |

## Reference

- [Models reference page](../../reference/models.md) for `ModelSpec`, `ModelFit`, and fit conventions.
