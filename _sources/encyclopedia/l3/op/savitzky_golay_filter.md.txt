# `savitzky_golay_filter` -- Polynomial-fit smoothing filter (Savitzky & Golay 1964).

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.savitzky_golay_transform`.

## Function signature

```python
mf.functions.savitzky_golay_transform(
    panel: pd.DataFrame,
    window: int,
    polyorder: int,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `window` | `int` | `7` | >= 3 | Length of the smoothing window. If even, rounded up to next odd integer (scipy requirement). |
| `polyorder` | `int` | `3` | — | Degree of the polynomial fit within each window. Must be < window. |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Local polynomial regression smoothing: each output value is the polynomial-fit centre value of a moving window. ``window_length`` (default 5) and ``polyorder`` (default 2) parameterise the kernel. Operational: runtime delegates to ``scipy.signal.savgol_filter`` (scipy is a hard dependency).

Used as the fixed-window baseline against which Goulet Coulombe & Klieber (2025) AlbaMA's adaptive-window estimator is compared in the v0.9.x replication recipe.

**When to use**

Smoothing macro indicator series for monitoring; AlbaMA replication baseline.

**When NOT to use**

Series with strong non-linear trends -- the polynomial fit smooths them out.

## In recipe context

Set ``params.op = "savitzky_golay_filter"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: savitzky_golay_filter
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'
* Savitzky & Golay (1964) 'Smoothing and Differentiation of Data by Simplified Least Squares Procedures', Analytical Chemistry 36(8).

## Related ops

See also: `hp_filter`, `hamilton_filter`, `ma_window`, `adaptive_ma_rf` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
