# `fourier` -- Fourier basis features -- sin/cos at fixed harmonics.

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.fourier_transform`.

## Function signature

```python
mf.functions.fourier_transform(
    panel: pd.DataFrame,
    n_terms: int,
    period: int,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `n_terms` | `int` | `4` | >= 1 | Number of harmonic pairs (sin + cos) to generate. Total output columns: 2 * n_terms. |
| `period` | `int` | `12` | >= 1 | Fundamental period of the seasonal pattern (e.g., 12 for monthly annual cycle, 4 for quarterly). |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Generates sin/cos pairs at harmonic frequencies of the calendar period (``params.period``, ``params.n_harmonics``). Captures smooth periodic patterns without the indicator-explosion of season_dummy.

**When to use**

Smooth seasonality (annual / weekly cycles) where dummies would over-fit.

## In recipe context

Set ``params.op = "fourier"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: fourier
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a pipeline of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

## Related ops

See also: `season_dummy`, `wavelet` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
