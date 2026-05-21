# `zscore_threshold` -- Flag observations beyond a z-score threshold.

[Back to `outlier_policy` axis](../axes/outlier_policy.md) | [Back to L2](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `outlier_policy`, sub-layer `l2_c`, layer `l2`.
> Standalone callable: `mf.functions.zscore_outlier_clean`.

## Function signature

```python
mf.functions.zscore_outlier_clean(
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

Computes the rolling z-score per series and flags ``|z|`` > ``leaf_config.zscore_threshold_value`` (default 3.0). Simpler than IQR but assumes approximately Gaussian residuals.

**When to use**

Approximately-Gaussian series; quick sanity-check sweeps.

## In recipe context

Set ``params.outlier_policy = "zscore_threshold"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L2 recipe fragment
params:
  outlier_policy: zscore_threshold
```

## References

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

## Related ops

See also: `mccracken_ng_iqr`, `winsorize` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
