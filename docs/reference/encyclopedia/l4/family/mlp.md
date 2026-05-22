# `mlp` -- Multi-layer perceptron (sklearn).

[Back to `family` axis](../axes/family.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `family`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.mlp_fit`.

## Function signature

```python
mf.functions.mlp_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> MLPFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |

## Returns

`MLPFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.n_params` | `int` | Total number of trainable parameters (weights + biases). |
| `.n_features_in_` | `int` | Number of input features seen during fit. |
| `.hidden_layer_sizes` | `tuple` | Tuple of hidden layer widths, e.g. (32, 16). |
| `.epochs_used` | `int` | Number of optimiser iterations completed. |
| `.final_loss` | `float` | Training MSE at the end of fitting. |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, shape (n_samples,). |
| `.summary()` | `str` | Arch metadata table: model_type, hidden_layer_sizes, n_features, n_params, epochs_used, final_loss. |

## Behavior

Feed-forward NN with ReLU activations. ``params.hidden_layer_sizes`` controls the architecture.

**v0.9 sub-axes** (apply equally to mlp / lstm / gru / transformer):
* ``params.architecture`` -- network topology. ``standard`` (default) is the standard feed-forward / sequence variant. ``hemisphere`` (future) implements Coulombe / Frenette / Klieber (2025 JAE) HNN with separate mean / variance hemispheres joined by a constraint loss.
* ``params.loss`` -- objective. ``mse`` (default), ``quantile`` (operational via forecast_object=quantile), ``volatility_emphasis`` (future, HNN constraint loss).

**When to use**

Non-linear regression baselines; ablations against deep NN.

## In recipe context

Set ``params.family = "mlp"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  family: mlp
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'

## Related ops

See also: `lstm`, `gru`, `transformer` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
