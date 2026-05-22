# `partial_least_squares` -- Partial least squares regression -- supervised factor extraction.

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.partial_least_squares_transform`.

## Function signature

```python
mf.functions.partial_least_squares_transform(
    panel: pd.DataFrame,
    target: pd.Series,
    n_components: int,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `target` | `pd.Series` | — | — | Supervisory signal aligned to the panel index. Must share at least one index value with panel; raises ValueError if the intersection is empty. |
| `n_components` | `int` | `3` | >= 1 | Number of PLS latent components. Clamped internally to min(T_clean - 1, K_clean). |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Computes orthogonal latent components that maximise the covariance with the target (not just predictor variance, as in PCA). sklearn's ``PLSRegression``; ``params.n_components``.

**When to use**

When a target-supervised reduction is preferable to PCA's unsupervised projection.

## In recipe context

Set ``params.op = "partial_least_squares"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: partial_least_squares
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'
* Wold, Sjöström & Eriksson (2001) 'PLS-regression: a basic tool of chemometrics', Chemometrics and Intelligent Laboratory Systems 58(2): 109-130.

## Related ops

See also: `pca`, `scaled_pca` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
