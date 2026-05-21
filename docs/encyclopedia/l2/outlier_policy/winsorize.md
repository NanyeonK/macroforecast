# `winsorize` -- Cap observations at user-supplied quantile thresholds (winsorization).

[Back to `outlier_policy` axis](../axes/outlier_policy.md) | [Back to L2](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `outlier_policy`, sub-layer `l2_c`, layer `l2`.
> Standalone callable: `mf.functions.winsorize_clean`.

## Function signature

```python
mf.functions.winsorize_clean(
    panel: pd.DataFrame,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | — |
| `lower_quantile` | `float` | `0.01` | — | — |
| `upper_quantile` | `float` | `0.99` | — | — |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

(See standalone callable docstring.)

## In recipe context

Set ``params.outlier_policy = "winsorize"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L2 recipe fragment
params:
  outlier_policy: winsorize
```

## References

* macroforecast design, L2: see design docs for winsorize.

## Related ops

See also: `mccracken_ng_iqr`, `zscore_threshold` (on the same axis).

_Last reviewed 2026-05-22 by macroforecast author._
