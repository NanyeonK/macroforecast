# `transformer` -- Transformer encoder regressor (torch, optional).

[Back to `family` axis](../axes/family.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `family`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.transformer_fit`.

## Function signature

```python
mf.functions.transformer_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> TransformerFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |

## Returns

`TransformerFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.n_params` | `int` | Total trainable parameters in Transformer encoder + head. |
| `.n_features_in_` | `int` | Number of input features seen during fit (= d_model). |
| `.hidden_size` | `int` | dim_feedforward of the single TransformerEncoderLayer. |
| `.epochs_used` | `int` | Number of training epochs completed. |
| `.final_loss` | `float` | Training MSE via no-grad forward pass after fitting. |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, shape (n_samples,). |
| `.summary()` | `str` | Arch metadata table: model_type, hidden_size, n_features, n_params, epochs_used, final_loss. |

## Behavior

Requires ``pip install macroforecast[deep]``. Self-attention on the lagged feature panel. Single encoder layer; suitable as a non-linear sequence-attention baseline.

**When to use**

Attention-based macro forecasting research; sequence-NN benchmark.

**When NOT to use**

Without [deep] installed.

## In recipe context

Set ``params.family = "transformer"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  family: transformer
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Vaswani et al. (2017) 'Attention is all you need', NeurIPS.

## Related ops

See also: `lstm`, `gru` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
