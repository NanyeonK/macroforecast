# Support vector

[Back to Models and Features](../model_overview.md)

Support vector regression fits a margin-based predictor and can use nonlinear kernels for flexible but controlled fits.

Pass any model string below as `Arm(model=...)`. Extra names an optional dependency, Scaling flags whether predictors should be standardized, and Tunable counts the hyperparameters the search space exposes.

| Model string | Description | Input | Extra | Scaling | Recommended preprocessing | Tunable |
| --- | --- | --- | --- | --- | --- | --- |
| `linear_svr` | Linear support-vector regression. | supervised | none | yes | standardize predictors before fitting | 2 |
| `nu_svr` | Nu support-vector regression. | supervised | none | yes | standardize predictors before fitting | 3 |
| `svr` | Kernel support-vector regression. | supervised | none | yes | standardize predictors before fitting | 3 |

## Reference

- [Models reference page](../../reference/models.md) for `ModelSpec`, `ModelFit`, and fit conventions.
