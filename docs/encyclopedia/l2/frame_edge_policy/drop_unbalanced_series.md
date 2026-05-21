# `drop_unbalanced_series` -- Drop columns that contain any NaN (retain only fully-observed series).

[Back to `frame_edge_policy` axis](../axes/frame_edge_policy.md) | [Back to L2](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `frame_edge_policy`, sub-layer `l2_e`, layer `l2`.
> Standalone callable: `mf.functions.drop_unbalanced_series_clean`.

## Function signature

```python
mf.functions.drop_unbalanced_series_clean(
    panel: pd.DataFrame,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | — |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

(See standalone callable docstring.)

## In recipe context

Set ``params.frame_edge_policy = "drop_unbalanced_series"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L2 recipe fragment
params:
  frame_edge_policy: drop_unbalanced_series
```

## References

* macroforecast design, L2: see design docs for drop_unbalanced_series.

## Related ops

See also: `truncate_to_balanced`, `zero_fill_leading` (on the same axis).

_Last reviewed 2026-05-22 by macroforecast author._
