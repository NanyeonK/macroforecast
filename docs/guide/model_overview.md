# Models and Features

[Back to documentation home](../index.md)

An `Arm` is built from two kinds of block, a set of feature steps and a single
model. This page lists both. It starts with the feature engineering steps that
turn a cleaned panel into model inputs, then helps you choose a model. Each model
family links to a detail page generated from the registry, so the lists always
match the installed version.

## Feature engineering

Features are organized into five families that recur across the macro
forecasting literature. F is principal-component or sparse factors, X is raw
lagged series, MARX is the moving-average lag cross that is the standard macro
design, MAF is maximum-autocorrelation factors, and Level passes untransformed
level columns through unchanged. The [Features](concepts/features.md) page covers
them in full, and the [Feature Engineering reference](../reference/feature_engineering.md)
lists every step parameter.

You compose these families from the step builders below. Each returns a step you
place in the `feature_steps` list of `feature_spec`, and stateful steps such as
PCA are refit inside every training window.

| Step builder | Description |
| --- | --- |
| `fourier_step` | Fourier seasonal-term step. |
| `group_pca_step` | Grouped-PCA step. |
| `hamilton_step` | Hamilton-filter step. |
| `interaction_step` | Pure-interaction step. |
| `lag_step` | Lag step. |
| `maf_step` | MAF step. |
| `marx_step` | MARX step. |
| `midas_step` | Fit linear MIDAS using equal step-function lag buckets. |
| `moving_average_step` | Moving-average-ladder step. |
| `nystroem_step` | Nystroem kernel-approximation step. |
| `partial_least_squares_step` | PLS step. |
| `pca_step` | PCA step. |
| `polynomial_step` | Polynomial-expansion step. |
| `random_projection_step` | Gaussian random-projection step. |
| `rolling_step` | Rolling-mean step. |
| `scale_step` | Scaling step. |
| `season_dummy_step` | Date-index seasonal-dummy step. |
| `seasonal_lag_step` | Seasonal-lag step. |
| `sliced_inverse_regression_step` | SIR step. |
| `sparse_pca_chen_rohe_step` | Chen-Rohe sparse component step. |
| `time_step` | Deterministic trend/month/quarter/year step. |
| `transform_step` | Deterministic column transform step. |
| `varimax_step` | Orthogonal varimax-rotation step. |

## Choosing a model

**Few predictors and a mostly linear signal.** Start from the benchmarks `ar`,
`ols`, `arima`, and the target-only `ucsv` inflation benchmark. They anchor any
comparison and are cheap to refit at every origin except for MCMC-based UCSV.

**Many predictors.** Regularize. `ridge` shrinks all coefficients, `lasso` and
`elastic_net` also select variables, and `adaptive_lasso` and `group_lasso` add
structured selection across feature blocks. For forecast averaging over many
candidate regressions, use `csr` or `jma`.

**Latent factor structure.** When series move together, extract common factors
with `far` and `favar`, or use a dynamic factor model from the mixed-frequency
family.

**Nonlinearity and interactions.** The macro forecasting literature finds this is
where the largest gains appear. The workhorses are `random_forest`, `xgboost`,
`lightgbm`, and the macro-adapted `macro_random_forest`.

**Sequence structure.** The neural family includes `lstm` and `gru` for recurrent
dynamics in longer panels.

**Conditional volatility.** For variance forecasting use `garch11`, `egarch`,
`gjr_garch`, and `realized_garch`.

## Model families

Each family has a detail page with a per-model table of inputs, optional dependencies, scaling, recommended preprocessing, and tunable counts.

- [Linear and regularized](models/linear.md) — 16 models
- [Factor models](models/factor.md) — 2 models
- [Classical time series](models/timeseries.md) — 15 models
- [Bayesian state-space](models/bayesian.md) — 1 models
- [Model averaging](models/model_averaging.md) — 2 models
- [Tree ensembles](models/tree.md) — 11 models
- [Support vector](models/support_vector.md) — 3 models
- [Nonparametric](models/nonparametric.md) — 2 models
- [Neural networks](models/neural.md) — 6 models
- [Volatility and GARCH](models/volatility.md) — 5 models
- [Mixed frequency](models/mixed_frequency.md) — 7 models
- [Assemblage](models/assemblage.md) — 6 models
- [Composite](models/composite.md) — 5 models
- [Spline](models/spline.md) — 1 models

## Notes

Feature steps are passed in the `feature_steps` list of
`mf.feature_engineering.feature_spec(...)`. Model strings are passed as the
`model` argument to `Arm(model=...)` or to `mf.forecasting.run(data, model=...)`.
Full feature-step parameters are on the
[Feature Engineering reference page](../reference/feature_engineering.md), and
model search spaces and presets are on the
[Models reference page](../reference/models.md) and, for fit-time ensembles, the
[Model Ensemble reference page](../reference/model_ensemble.md). The generated
[Model x Forecast Policy Matrix](model_policy_matrix.md) states which forecast
policies are supported for each registered model.

```{toctree}
:hidden:
:maxdepth: 1

models/linear
models/factor
models/timeseries
models/bayesian
models/model_averaging
models/tree
models/support_vector
models/nonparametric
models/neural
models/volatility
models/mixed_frequency
models/assemblage
models/composite
models/spline
model_policy_matrix
```
