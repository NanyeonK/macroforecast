# `lstm` -- Long short-term memory recurrent NN (torch, optional).

[Back to `family` axis](../axes/family.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `family`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.lstm_fit`.

## Function signature

```python
mf.functions.lstm_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> LSTMFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |

## Returns

`LSTMFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.n_params` | `int` | Total number of trainable parameters in LSTM + head. |
| `.n_features_in_` | `int` | Number of input features seen during fit. |
| `.hidden_size` | `int` | Width of the LSTM hidden state. |
| `.epochs_used` | `int` | Number of training epochs completed. |
| `.final_loss` | `float` | Training MSE via no-grad forward pass after fitting. |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, shape (n_samples,). |
| `.summary()` | `str` | Arch metadata table: model_type, hidden_size, n_features, n_params, epochs_used, final_loss. |

## Behavior

Requires ``pip install macroforecast[deep]``. Sequence-aware RNN with input/forget/output gates. Trains on sliding windows of the lagged feature panel.

**When to use**

Sequence-modelling studies; replication of deep-NN forecasting papers.

**When NOT to use**

Without [deep] installed -- raises NotImplementedError.

## In recipe context

Set ``params.family = "lstm"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  family: lstm
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Hochreiter & Schmidhuber (1997) 'Long short-term memory', Neural Computation 9(8).

## Related ops

See also: `gru`, `transformer`, `mlp` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
