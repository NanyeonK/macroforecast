# `interaction` -- Pairwise interaction terms only (no pure powers).

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.interaction_terms_transform`.

## Function signature

```python
mf.functions.interaction_terms_transform(
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

Subset of polynomial degree-2 features that contains only pairwise products ``x_i · x_j`` for ``i ≠ j``. Cheaper than full polynomial expansion when interaction structure (not non-linearity in single inputs) is the target.

**When to use**

Capturing predictor-pair complementarities in linear models.

## In recipe context

Set ``params.op = "interaction"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: interaction
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

## Related ops

See also: `polynomial` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
