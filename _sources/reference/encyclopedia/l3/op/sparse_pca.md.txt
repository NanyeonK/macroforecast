# `sparse_pca` -- Sparse PCA -- L1-penalised factor loadings (sklearn / Zou-Hastie-Tibshirani 2006).

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.sparse_pca_transform`.

## Function signature

```python
mf.functions.sparse_pca_transform(
    panel: pd.DataFrame,
    n_components: int,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `n_components` | `int` | `8` | >= 1 | Number of sparse principal components to extract. Clamped internally to min(T_clean, K) - 1. |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Variant of PCA where loadings are pushed toward zero by an L1 penalty (``params.alpha``). Yields more interpretable factors at the cost of a small reconstruction loss; uses sklearn's ``SparsePCA``.

**When to use**

When you want factor loadings to map cleanly onto a small subset of original predictors (interpretability).

**When NOT to use**

When pure variance maximisation is more important than interpretability -- use plain ``pca``. For the Chen-Rohe (2023) SCA variant used in Zhou-Rapach (2025) Sparse Macro-Finance Factors, use ``sparse_pca_chen_rohe`` instead.

## In recipe context

Set ``params.op = "sparse_pca"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: sparse_pca
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

## Related ops

See also: `pca`, `scaled_pca`, `sparse_pca_chen_rohe`, `supervised_pca` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
