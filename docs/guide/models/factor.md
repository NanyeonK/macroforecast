# Factor models

[Back to Models and Features](../model_overview.md)

Factor models summarize many comoving series into a few latent factors and forecast from those, which suits the strong common movement in macro panels.

Pass any model string below as `Arm(model=...)`. Extra names an optional dependency, Scaling flags whether predictors should be standardized, and Tunable counts the hyperparameters the search space exposes.

| Model string | Description | Input | Extra | Scaling | Recommended preprocessing | Tunable |
| --- | --- | --- | --- | --- | --- | --- |
| `far` | Factor-augmented autoregression. | supervised | none | no | default | 2 |
| `favar` | FAVAR::FAVAR-aligned Bayesian factor-augmented VAR sampler. | supervised | none | no | default | 2 |

## Reference

- [Models reference page](../../reference/models.md) for `ModelSpec`, `ModelFit`, and fit conventions.
