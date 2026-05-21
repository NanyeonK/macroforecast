# `pca` -- Principal component analysis -- linear factor extraction.

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.pca_transform`.

## Function signature

```python
mf.functions.pca_transform(
    panel: pd.DataFrame,
    n_components: int | str,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `n_components` | `int | str` | `3` | >= 1 or 'all' | Number of principal components to extract. Clamped to min(T, K) - 1 internally. Sentinel 'all' extracts full effective rank min(T, K). |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Eigendecomposition of the column covariance; returns the top ``params.n_components`` principal components. Implements the Stock-Watson (2002) diffusion-index workflow used throughout FRED-MD applications.

Combine with ``factor_augmented_ar`` or ``factor_augmented_var`` at L4 to build the diffusion-index forecaster. ``temporal_rule`` controls whether components are re-fit per origin (default: ``expanding_window_per_origin``).

**When to use**

Reducing FRED-MD's 100+ predictors to a handful of latent factors; factor-augmented forecasts.

## In recipe context

Set ``params.op = "pca"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: pca
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'
* Stock & Watson (2002) 'Forecasting Using Principal Components from a Large Number of Predictors', JASA 97(460): 1167-1179.

## Related ops

See also: `sparse_pca`, `scaled_pca`, `varimax`, `dfm`, `partial_least_squares` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
