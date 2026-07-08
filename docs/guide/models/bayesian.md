# Bayesian state-space

[Back to Models and Features](../model_overview.md)

Bayesian state-space models estimate latent target components and posterior forecast summaries; UCSV is the canonical inflation trend benchmark with stochastic volatility.

Pass any model string below as `Arm(model=...)`. Extra names an optional dependency, Scaling flags whether predictors should be standardized, and Tunable counts the hyperparameters the search space exposes.

| Model string | Description | Input | Extra | Scaling | Recommended preprocessing | Tunable |
| --- | --- | --- | --- | --- | --- | --- |
| `ucsv` | Stock-Watson UCSV target-only benchmark with horizon-invariant final-trend forecasts. | target | none | no | default | 0 |

## Reference

- [Models reference page](../../reference/models.md) for `ModelSpec`, `ModelFit`, and fit conventions.
