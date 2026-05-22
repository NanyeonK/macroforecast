# `ma_increasing_order` -- MARX -- moving averages of increasing order (Coulombe 2024).

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.ma_increasing_order_transform`.

## Function signature

```python
mf.functions.ma_increasing_order_transform(
    panel: pd.DataFrame,
    max_order: int,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `max_order` | `int` | `12` | >= 2 | Maximum window order. Generates windows 2, 3, ..., max_order. |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Stacks moving averages with windows ``[1, 2, 4, 8, ..., w_max]`` into a multi-column block. Captures multi-scale persistence in a single op; popular feature in macroeconomic random forest pipelines.

Implements the MARX (Moving-Average-of-Random-eXogeneous) trick from Coulombe (2024).

**When to use**

Tree / RF models that benefit from multi-scale temporal features without manual lag selection.

## In recipe context

Set ``params.op = "ma_increasing_order"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: ma_increasing_order
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'
* Coulombe (2024) 'The Macroeconomic Random Forest', Journal of Applied Econometrics 39(7): 1190-1209.

## Related ops

See also: `ma_window`, `lag` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
