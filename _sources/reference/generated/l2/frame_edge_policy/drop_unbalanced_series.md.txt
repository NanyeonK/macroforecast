# `drop_unbalanced_series` -- Drop predictor columns that aren't observed across the full sample.

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
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Trades predictor count for sample length. Useful when the recipe wants to keep early observations and is willing to lose late-arrival series.

**When to use**

Long-history studies (1959-) where late-introduction series should be excluded.

## In recipe context

Set ``params.frame_edge_policy = "drop_unbalanced_series"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L2 recipe fragment
params:
  frame_edge_policy: drop_unbalanced_series
```

## References

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

## Related ops

See also: `truncate_to_balanced`, `keep_unbalanced` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
