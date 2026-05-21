# `zscore_threshold` -- Flag or replace outliers beyond a z-score threshold.

[Back to `outlier_policy` axis](../axes/outlier_policy.md) | [Back to L2](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `outlier_policy`, sub-layer `l2_c`, layer `l2`.
> Standalone callable: `mf.functions.zscore_outlier_clean`.

## Function signature

```python
mf.functions.zscore_outlier_clean(
    panel: pd.DataFrame,
    threshold: float = 3.0,
    action: str = 'flag_as_nan',
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | — |
| `threshold` | `float` | `3.0` | — | — |
| `action` | `str` | `'flag_as_nan'` | — | — |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

(See standalone callable docstring.)

## In recipe context

Set ``params.outlier_policy = "zscore_threshold"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L2 recipe fragment
params:
  outlier_policy: zscore_threshold
```

## References

* macroforecast design, L2: see design docs for zscore_threshold.

## Related ops

See also: `mccracken_ng_iqr`, `winsorize` (on the same axis).

_Last reviewed 2026-05-22 by macroforecast author._
