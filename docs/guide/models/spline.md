# Spline

[Back to Models and Features](../model_overview.md)

Spline models fit smooth nonlinear functions of the predictors using basis expansions.

Pass any model string below as `Arm(model=...)`. Extra names an optional dependency, Scaling flags whether predictors should be standardized, and Tunable counts the hyperparameters the search space exposes.

| Model string | Description | Input | Extra | Scaling | Recommended preprocessing | Tunable |
| --- | --- | --- | --- | --- | --- | --- |
| `mars` | Package-native MARS-style hinge-basis regression. | supervised | none | no | default | 3 |

## Reference

- [Models reference page](../../reference/models.md) for `ModelSpec`, `ModelFit`, and fit conventions.
