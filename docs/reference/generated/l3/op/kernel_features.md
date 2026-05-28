# `kernel_features` -- Random Fourier features -- approximate RBF kernel via random projection.

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.kernel_features_transform`.

## Function signature

```python
mf.functions.kernel_features_transform(
    panel: pd.DataFrame,
    kind: str enum {"rbf", "polynomial"},
    gamma: float,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `kind` | `str enum {"rbf", "polynomial"}` | `'"rbf"'` | — | Kernel type. 'rbf' for Gaussian kernel; 'polynomial' for degree-2 polynomial kernel. |
| `gamma` | `float` | `1.0` | > 0 | Kernel bandwidth. For rbf: exp(-gamma * ||x-z||^2). For polynomial: (gamma * <x,z> + 1)^2. |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

sklearn ``RBFSampler``: maps inputs to ``params.n_components`` random Fourier features whose dot product approximates the RBF kernel. Enables linear models to fit RBF-kernelised responses at training-set-size linear cost.

**When to use**

Kernel-augmented ridge / SVM at scale (n > 10k).

**When NOT to use**

Small-sample problems where exact kernel SVM is feasible.

## In recipe context

Set ``params.op = "kernel_features"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: kernel_features
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a pipeline of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'
* Rahimi & Recht (2007) 'Random Features for Large-Scale Kernel Machines', NeurIPS.

## Related ops

See also: `kernel`, `nystroem` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
