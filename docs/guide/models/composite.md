# Composite

[Back to Models and Features](../model_overview.md)

Composite models combine several base learners inside one fit.

Pass any model string below as `Arm(model=...)`. Extra names an optional dependency, Scaling flags whether predictors should be standardized, and Tunable counts the hyperparameters the search space exposes.

| Model string | Description | Input | Extra | Scaling | Recommended preprocessing | Tunable |
| --- | --- | --- | --- | --- | --- | --- |
| `pcr` | Principal component regression with optional control residualization. | supervised | none | no | default | 1 |
| `pls` | Partial least squares regression with optional control residualization. | supervised | none | no | default | 1 |
| `scaled_pca` | Huang et al. scaled PCA: marginal predictive-slope scaling followed by PCA. | supervised | none | no | default | 1 |
| `supervised_pca` | Original-style iterative supervised PCA with residual correlation screening and projection. | supervised | none | no | default | 3 |
| `supervised_scaled_pca` | Hounyo-Li supervised scaled PCA: marginal predictive-slope scaling followed by SPCA. | supervised | none | no | default | 3 |

## Reference

- [Models reference page](../../reference/models.md) for `ModelSpec`, `ModelFit`, and fit conventions.
