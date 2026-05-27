# `lag` -- Lagged target/predictor block.

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.lag_matrix`.

## Function signature

```python
mf.functions.lag_matrix(
    panel: pd.DataFrame,
    n_lag: int,
    include_contemporaneous: bool,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `n_lag` | `int` | `4` | >= 1 | Number of lags. Output has K * n_lag columns. |
| `include_contemporaneous` | `bool` | `False` | — | If True, also include lag 0 (the contemporaneous column). |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Constructs a lagged matrix from inputs. ``params.n_lag`` sets the lag depth. Standard predictor for autoregressive baselines.

**When to use**

Always when building AR features or lagged-X feature blocks.

**When NOT to use**

When the target itself is already differenced/lagged in L2 -- avoid double-lagging.

## In recipe context

Set ``params.op = "lag"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: lag
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a pipeline of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

## Related ops

See also: `seasonal_lag`, `target_construction` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
