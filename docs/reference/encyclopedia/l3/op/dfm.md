# `dfm` -- Dynamic factor model -- Kalman state-space factor extraction.

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.dfm_transform`.

## Function signature

```python
mf.functions.dfm_transform(
    panel: pd.DataFrame,
    n_factors: int,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `n_factors` | `int` | `3` | >= 1 | Number of latent dynamic factors to extract. Clamped internally to min(T_clean, K) - 1. |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

statsmodels ``DynamicFactor`` MLE estimate of latent factors with idiosyncratic AR(1) errors. Differs from ``pca`` in that factors are smoothed via the Kalman filter and respect a factor-VAR transition.

When the panel is mixed-frequency (FRED-SD), the runtime auto-routes to ``DynamicFactorMQ`` (Mariano-Murasawa 2003).

**When to use**

Smoothed factors with an explicit dynamic; mixed-frequency panels (FRED-SD).

## In recipe context

Set ``params.op = "dfm"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: dfm
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'
* Mariano & Murasawa (2003) 'A new coincident index of business cycles based on monthly and quarterly series', JAE 18(4): 427-443.

## Related ops

See also: `pca`, `scaled_pca` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
