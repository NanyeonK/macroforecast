# Linear and regularized

[Back to Models and Features](../model_overview.md)

Linear and regularized models predict with a weighted sum of features. They run from ordinary least squares through ridge, lasso, and elastic net shrinkage to structured and adaptive penalties that also select variables.

Pass any model string below as `Arm(model=...)`. Extra names an optional dependency, Scaling flags whether predictors should be standardized, and Tunable counts the hyperparameters the search space exposes.

| Model string | Description | Input | Extra | Scaling | Recommended preprocessing | Tunable |
| --- | --- | --- | --- | --- | --- | --- |
| `adaptive_elastic_net` | Adaptive elastic net using initial coefficient-based column weights. | supervised | none | no | default | 3 |
| `adaptive_lasso` | Adaptive lasso using initial coefficient-based penalty weights. | supervised | none | no | default | 2 |
| `bayesian_ridge` | Empirical-Bayes Bayesian ridge. | supervised | none | no | default | 0 |
| `elastic_net` | Elastic net regression. | supervised | none | no | default | 2 |
| `fused_difference_ridge` | Ridge regression with a fused-difference coefficient prior. | supervised | none | no | default | 1 |
| `glmboost` | Componentwise linear boosting. | supervised | none | no | default | 2 |
| `group_lasso` | Package-native group lasso with group-level sparsity. | supervised | none | no | default | 1 |
| `huber` | Robust Huber regression. | supervised | none | no | default | 1 |
| `lasso` | Lasso regression. | supervised | none | no | default | 1 |
| `nonneg_ridge` | Ridge regression with non-negative coefficients. | supervised | none | no | default | 1 |
| `ols` | Ordinary least squares with no model-owned tuning space. | supervised | none | no | default | 0 |
| `random_walk_ridge` | Time-varying random-walk ridge fit, predicting with the final coefficient vector. | supervised | none | no | default | 1 |
| `ridge` | Ridge regression. | supervised | none | no | default | 1 |
| `shrink_to_target_ridge` | Ridge regression shrinking coefficients toward a target vector. | supervised | none | no | default | 1 |
| `sparse_group_lasso` | Package-native sparse group lasso with group and feature-level sparsity. | supervised | none | no | default | 2 |
| `tvp_ridge` | Goulet Coulombe TVP ridge / 2SRR estimator. | supervised | none | no | default | 1 |

## Reference

- [Models reference page](../../reference/models.md) for `ModelSpec`, `ModelFit`, and fit conventions.
