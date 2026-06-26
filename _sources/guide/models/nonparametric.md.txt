# Nonparametric

[Back to Models and Features](../model_overview.md)

Nonparametric models make few functional-form assumptions and let the data shape the fit.

Pass any model string below as `Arm(model=...)`. Extra names an optional dependency, Scaling flags whether predictors should be standardized, and Tunable counts the hyperparameters the search space exposes.

| Model string | Description | Input | Extra | Scaling | Recommended preprocessing | Tunable |
| --- | --- | --- | --- | --- | --- | --- |
| `kernel_ridge` | Kernel ridge regression. | supervised | none | yes | standardize predictors before nonlinear kernels | 2 |
| `knn` | K-nearest-neighbor regression. | supervised | none | yes | standardize predictors before distance-based fitting | 2 |

## Reference

- [Models reference page](../../reference/models.md) for `ModelSpec`, `ModelFit`, and fit conventions.
