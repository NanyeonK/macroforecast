# `gru` -- Gated recurrent unit RNN (torch, optional).

[Back to `family` axis](../axes/family.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `family`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.gru_fit`.

## Function signature

```python
mf.functions.gru_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> GRUFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |

## Returns

`GRUFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.n_params` | `int` | Total number of trainable parameters in GRU + head. |
| `.n_features_in_` | `int` | Number of input features seen during fit. |
| `.hidden_size` | `int` | Width of the GRU hidden state. |
| `.epochs_used` | `int` | Number of training epochs completed. |
| `.final_loss` | `float` | Training MSE via no-grad forward pass after fitting. |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, shape (n_samples,). |
| `.summary()` | `str` | Arch metadata table: model_type, hidden_size, n_features, n_params, epochs_used, final_loss. |

## Behavior

Requires ``pip install macroforecast[deep]``. Simpler than LSTM (one fewer gate); often comparable on macro panels.

**When to use**

Sequence-modelling baselines; LSTM ablations.

**When NOT to use**

Without [deep] installed.

## In recipe context

Set ``params.family = "gru"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  family: gru
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Cho et al. (2014) 'Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation', EMNLP.

## Related ops

See also: `lstm`, `transformer` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
