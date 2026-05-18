# `wavelet` -- Discrete wavelet transform -- multi-scale time-frequency features.

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.wavelet_transform`.

## Function signature

```python
mf.functions.wavelet_transform(
    panel: pd.DataFrame,
    wavelet: str,
    n_levels: int,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `wavelet` | `str` | `'"db4"'` | — | Wavelet family name (e.g., "db4", "haar"). Accepted for API consistency; runtime uses a rolling-mean low-pass approximation. |
| `n_levels` | `int` | `3` | >= 1 | Number of decomposition levels. Each level produces an approximation (_wA{level}) and detail (_wD{level}) pair. |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Decomposes the series into wavelet detail and approximation coefficients at several scales (``params.wavelet``, ``params.level``). Captures localised time-frequency patterns that Fourier basis cannot.

**When to use**

Series with localised oscillations or non-stationary cycles (financial / climate macro).

**When NOT to use**

Smooth seasonal patterns -- use ``fourier`` instead.

## In recipe context

Set ``params.op = "wavelet"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: wavelet
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

## Related ops

See also: `fourier` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
