# `asymmetric_trim` -- Albacore-family rank-space transformation (Goulet Coulombe et al. 2024).

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.asymmetric_trim_transform`.

## Function signature

```python
mf.functions.asymmetric_trim_transform(
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

Per-period sort: panel ``Π`` of shape ``(T, K)`` is mapped to ``O`` where ``O[t, r] = sort(Π[t, :])[r]`` (ascending). Asymmetric trimming emerges in the *downstream* nonneg ridge (``ridge(coefficient_constraint=nonneg)``) that learns rank-position weights -- this op does the rank-space transformation only.

Optional ``smooth_window > 0`` applies a centred moving average to each rank-position time series (paper §3 mentions 3-month MA for noisy components; users can chain ``ma_window`` explicitly when they want a different window).

Operational from v0.8.9 (B-6). Layer scope ``(preprocessing, l3)`` so the L3 pipeline can dispatch it at recipe time. Algorithm spec: ``docs/replications/maximally_forward_looking_algorithm_notes.md``.

**When to use**

Building Albacore_ranks-style core inflation indicators; supervised asymmetric trimming where the band is learned from data.

**When NOT to use**

Symmetric trimmed-mean targets (use a fixed-window ``ma_window`` instead).

## In recipe context

Set ``params.op = "asymmetric_trim"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: asymmetric_trim
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a pipeline of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'
* Goulet Coulombe, Klieber, Barrette & Goebel (2024) 'Maximally Forward-Looking Core Inflation', technical report (R package: assemblage).

## Related ops

See also: `ma_window`, `ma_increasing_order`, `scaled_pca` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
