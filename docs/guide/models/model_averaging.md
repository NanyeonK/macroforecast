# Model averaging

[Back to Models and Features](../model_overview.md)

Model averaging methods combine many candidate regressions, either by averaging complete subsets or by optimizing cross-validation weights.

Pass any model string below as `Arm(model=...)`. Extra names an optional dependency, Scaling flags whether predictors should be standardized, and Tunable counts the hyperparameters the search space exposes.

| Model string | Description | Input | Extra | Scaling | Recommended preprocessing | Tunable |
| --- | --- | --- | --- | --- | --- | --- |
| `csr` | Complete Subset Regression; averages OLS forecasts over k-predictor subsets. | supervised | none | no | default | 1 |
| `jma` | Jackknife Model Averaging with simplex weights chosen by OLS leave-one-out CV. | supervised | none | no | default | 0 |

## Reference

- [Models reference page](../../reference/models.md) for `ModelSpec`, `ModelFit`, and fit conventions.
