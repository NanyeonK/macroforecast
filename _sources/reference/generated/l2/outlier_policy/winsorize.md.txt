# `winsorize` -- Cap observations at user-supplied quantile thresholds.

[Back to `outlier_policy` axis](../axes/outlier_policy.md) | [Back to L2](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `outlier_policy`, sub-layer `l2_c`, layer `l2`.
> Standalone callable: `mf.functions.winsorize_clean`.

## Function signature

```python
mf.functions.winsorize_clean(
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

Truncates each series at ``leaf_config.winsorize_lower_quantile`` (default 0.01) and ``leaf_config.winsorize_upper_quantile`` (default 0.99). Less aggressive than the McCracken-Ng IQR rule and preserves more of the tail.

**When to use**

Studies that want a bounded but non-NaN outlier handler; alternative-rule comparisons.

## In recipe context

Set ``params.outlier_policy = "winsorize"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L2 recipe fragment
params:
  outlier_policy: winsorize
```

## References

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'
* Tukey (1977) 'Exploratory Data Analysis', Addison-Wesley.

## Related ops

See also: `mccracken_ng_iqr`, `zscore_threshold` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
