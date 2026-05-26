# `lstm_hidden_state` -- LSTM hidden-state activation heatmap (Karpathy et al. 2015).

[Back to `op` axis](../axes/op.md) | [Back to L7](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L7_A_importance_dag_body`, layer `l7`.
> Standalone callable: `mf.functions.lstm_hidden_state`.

## Function signature

```python
mf.functions.lstm_hidden_state(...)
```

## Behavior

Extracts the per-timestep hidden-state activations ``h_t`` from a fitted LSTM model and renders them as a heatmap: rows index hidden units, columns index observations, color encodes mean ``|h_t|``. The visualization follows the approach of Karpathy, Johnson & Fei-Fei (2015) for understanding which hidden units fire on which parts of the input sequence.

Implementation details:

* Requires ``macroforecast[deep]`` (PyTorch); a ``NotImplementedError`` is raised for non-torch models (``transformer``, ``gru`` without ``torch``, etc.).
* State capture uses a forward hook registered on the LSTM layer via ``torch.nn.Module.register_forward_hook``.
* Output: a DataFrame of shape ``(n_hidden_units, T)`` where ``T`` is the sequence length. Column names are observation dates when the input panel carries a DatetimeIndex; otherwise integer positions.
* The ``l7_importance_v1`` sink carries the heatmap frame under the key ``hidden_state_activations``.

Only compatible with the ``lstm`` model in L4. The ``transformer`` model raises ``NotImplementedError`` because it has no recurrent hidden state; use ``attention_weights`` for transformer attribution.

**When to use**

Diagnosing LSTM sequence dynamics; identifying which hidden units respond to macro regime shifts.

**When NOT to use**

Non-LSTM models -- use attention_weights for transformers, shap_deep / gradient_shap / integrated_gradients for general deep-learning attribution.

## In recipe context

Set ``params.op = "lstm_hidden_state"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L7 recipe fragment
params:
  op: lstm_hidden_state
```

## References

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Karpathy, Johnson & Fei-Fei (2015) 'Visualizing and Understanding Recurrent Networks', arXiv:1506.02078. <https://arxiv.org/abs/1506.02078>

## Related ops

See also: `attention_weights`, `shap_deep`, `gradient_shap`, `integrated_gradients` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
