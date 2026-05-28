# `truncate_to_balanced` -- Trim leading / trailing rows until every series is observed.

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
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Makes the panel rectangular by removing rows where any predictor (or the target, depending on scope) is missing. Standard for factor-model-style studies that need a balanced panel.

**When to use**

Default for high-dimensional studies; pairs with em_factor imputation for the interior.

## In recipe context

Set ``params.frame_edge_policy = "truncate_to_balanced"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L2 recipe fragment
params:
  frame_edge_policy: truncate_to_balanced
```

## References

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'
* Stock & Watson (2002) 'Macroeconomic Forecasting Using Diffusion Indexes', JBES 20(2).

## Related ops

See also: `drop_unbalanced_series`, `keep_unbalanced`, `zero_fill_leading` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
