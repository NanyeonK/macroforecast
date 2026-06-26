# Assemblage

[Back to Models and Features](../model_overview.md)

Assemblage models aggregate many component predictions or ranks into a single forecast.

Pass any model string below as `Arm(model=...)`. Extra names an optional dependency, Scaling flags whether predictors should be standardized, and Tunable counts the hyperparameters the search space exposes.

| Model string | Description | Input | Extra | Scaling | Recommended preprocessing | Tunable |
| --- | --- | --- | --- | --- | --- | --- |
| `albacore_components` | Inflation-specific component-space Albacore wrapper. | supervised | none | no | default | 1 |
| `albacore_ranks` | Inflation-specific rank-space Albacore wrapper. | supervised | none | no | default | 1 |
| `assemblage_regression` | Generic assemblage regression wrapper with component and rank variants. | supervised | none | no | default | 1 |
| `component_aggregation` | Component-space supervised aggregation; generic Albacorecomps primitive. | supervised | none | no | default | 1 |
| `rank_aggregation` | Rank-space supervised aggregation; generic Albacoreranks primitive. | supervised | none | no | default | 1 |
| `supervised_aggregation` | Generic constrained supervised aggregation derived from Albacore/assemblage primitives. | supervised | none | no | default | 1 |

## Reference

- [Models reference page](../../reference/models.md) for `ModelSpec`, `ModelFit`, and fit conventions.
