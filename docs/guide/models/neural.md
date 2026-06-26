# Neural networks

[Back to Models and Features](../model_overview.md)

Neural networks learn flexible nonlinear maps, including recurrent forms for sequence structure in longer panels.

Pass any model string below as `Arm(model=...)`. Extra names an optional dependency, Scaling flags whether predictors should be standardized, and Tunable counts the hyperparameters the search space exposes.

| Model string | Description | Input | Extra | Scaling | Recommended preprocessing | Tunable |
| --- | --- | --- | --- | --- | --- | --- |
| `density_hnn` | Paper-faithful Density Hemisphere neural network with prior-DNN OOB volatility emphasis and OOB volatility rescaling. | supervised | `deep` | no | feature lags/trends are built before fitting; X and y are standardized inside each fit | 4 |
| `gru` | Torch-backed GRU regressor. | supervised | `deep` | no | handled internally: X and y are standardized inside each fit | 3 |
| `hemisphere_nn` | Bagged Hemisphere neural network with mean and variance heads. | supervised | `deep` | no | handled internally: X is standardized inside each fit | 3 |
| `lstm` | Torch-backed LSTM regressor. | supervised | `deep` | no | handled internally: X and y are standardized inside each fit | 3 |
| `nn` | Torch-backed feed-forward multilayer perceptron regressor. | supervised | `deep` | no | handled internally: X and y are standardized inside each fit | 4 |
| `transformer` | Torch-backed Transformer encoder regressor. | supervised | `deep` | no | handled internally: X and y are standardized inside each fit | 3 |

## Reference

- [Models reference page](../../reference/models.md) for `ModelSpec`, `ModelFit`, and fit conventions.
