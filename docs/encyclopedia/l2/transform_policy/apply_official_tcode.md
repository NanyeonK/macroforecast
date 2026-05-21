# `apply_official_tcode` -- Apply McCracken-Ng t-code stationarity transforms per column.

[Back to `transform_policy` axis](../axes/transform_policy.md) | [Back to L2](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `transform_policy`, sub-layer `l2_b`, layer `l2`.
> Standalone callable: `mf.functions.apply_tcode_transform`.

## Function signature

```python
mf.functions.apply_tcode_transform(
    panel: pd.DataFrame,
    tcode_map: dict[str, int],
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | — |
| `tcode_map` | `dict[str, int]` | — | — | — |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

(See standalone callable docstring.)

## In recipe context

Set ``params.transform_policy = "apply_official_tcode"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L2 recipe fragment
params:
  transform_policy: apply_official_tcode
```

## References

* macroforecast design, L2: see design docs for apply_official_tcode.

## Related ops

See also: `asymmetric_trim`, `custom_tcode`, `diff`, `level`, `log` (on the same axis).

_Last reviewed 2026-05-22 by macroforecast author._
