# Tree ensembles

[Back to Models and Features](../model_overview.md)

Tree ensembles average or boost many decision trees. They capture nonlinearity and interactions automatically and are the workhorses behind the largest reported macro forecasting gains.

Pass any model string below as `Arm(model=...)`. Extra names an optional dependency, Scaling flags whether predictors should be standardized, and Tunable counts the hyperparameters the search space exposes.

| Model string | Description | Input | Extra | Scaling | Recommended preprocessing | Tunable |
| --- | --- | --- | --- | --- | --- | --- |
| `catboost` | CatBoost regressor. | supervised | `catboost` | no | default | 3 |
| `decision_tree` | CART regression tree. | supervised | none | no | default | 2 |
| `extra_trees` | Extremely randomized trees. | supervised | none | no | default | 3 |
| `gradient_boosting` | Gradient-boosted regression trees. | supervised | none | no | default | 3 |
| `lgb_plus` | LGB+ competition hybrid boosting with tree/linear channel diagnostics. | supervised | `lightgbm` | no | default | 7 |
| `lgba_plus` | LGB^A+ alternating tree-block and greedy linear boosting. | supervised | `lightgbm` | no | default | 7 |
| `lightgbm` | LightGBM regressor. | supervised | `lightgbm` | no | default | 4 |
| `macro_random_forest` | Adapter for the external MacroRandomForest package. | supervised | `macro_random_forest` | no | default | 8 |
| `quantile_regression_forest` | Quantile regression forest. | supervised | none | no | default | 3 |
| `random_forest` | Random forest regression. | supervised | none | no | default | 4 |
| `xgboost` | XGBoost regressor. | supervised | `xgboost` | no | default | 4 |

## Reference

- [Models reference page](../../reference/models.md) for `ModelSpec`, `ModelFit`, and fit conventions.
