# `polynomial` -- Polynomial basis expansion -- degree-d powers of input.

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.polynomial_expansion_transform`.

## Function signature

```python
mf.functions.polynomial_expansion_transform(
    panel: pd.DataFrame,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

sklearn ``PolynomialFeatures`` of degree ``params.degree``. Includes interaction terms by default; set ``params.interaction_only=True`` for products without pure powers.

**When to use**

Capturing low-order non-linearity for linear / kernel models.

**When NOT to use**

High dimension (degree > 3 with many predictors) -- explodes the design matrix; use kernel methods instead.

## In recipe context

Set ``params.op = "polynomial"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: polynomial
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

## Related ops

See also: `interaction`, `kernel_features`, `polynomial_expansion` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
