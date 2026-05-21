# `truncate_to_balanced` -- Remove rows containing any NaN to produce a balanced panel.

[Back to `frame_edge_policy` axis](../axes/frame_edge_policy.md) | [Back to L2](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `frame_edge_policy`, sub-layer `l2_e`, layer `l2`.
> Standalone callable: `mf.functions.truncate_to_balanced_clean`.

## Function signature

```python
mf.functions.truncate_to_balanced_clean(
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

Set ``params.frame_edge_policy = "truncate_to_balanced"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L2 recipe fragment
params:
  frame_edge_policy: truncate_to_balanced
```

## References

* macroforecast design, L2: see design docs for truncate_to_balanced.

## Related ops

See also: `drop_unbalanced_series`, `zero_fill_leading` (on the same axis).

_Last reviewed 2026-05-22 by macroforecast author._
