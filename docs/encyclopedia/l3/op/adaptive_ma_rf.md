# `adaptive_ma_rf` -- AlbaMA -- RF-driven adaptive moving average smoother for a single time series.

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.adaptive_ma_rf_transform`.

## Function signature

```python
mf.functions.adaptive_ma_rf_transform(
    panel: pd.DataFrame,
    n_estimators: int,
    min_samples_leaf: int,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `n_estimators` | `int` | `100` | >= 1 | Number of trees in the RF ensemble. Paper recommends 500; 100 is the default for speed. |
| `min_samples_leaf` | `int` | `40` | >= 1 | Minimum samples per leaf; lower-bounds the effective adaptive window length (paper default: 40). |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Goulet Coulombe & Klieber (2025) 'Adaptive Moving Average for Macroeconomic Monitoring' (arXiv:2501.13222 §2). A random forest fit with a *single* regressor -- the time index -- on the target series ``y`` (i.e. ``RF(y_t ~ t)``). Per-observation leaf membership induces a weight matrix ``w_τt`` whose row sums to 1, so the smoother is a learned-bandwidth moving average of ``y``; the realised window adapts to local volatility / regime. Paper p.8 defaults: ``n_estimators = B = 500``, ``min_samples_leaf = 40``, ``max_features = 1``. ``sided = 'two'`` (default) fits one forest on the full sample (retrospective smoother); ``sided = 'one'`` fits an expanding-window forest per ``t`` (real-time nowcasting variant, paper §3.3 / p.10).

Atomic primitive: existing ``ma_window`` uses a fixed length; ``hamilton_filter`` is a regression on lags rather than a moving average; neither composes into AlbaMA without a learned window selector.

**When to use**

Replicating AlbaMA recipes; macro indicator monitoring under regime shifts.

**When NOT to use**

Multivariate denoising of a predictor panel (AlbaMA smooths a single target series).

## In recipe context

Set ``params.op = "adaptive_ma_rf"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: adaptive_ma_rf
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'
* Goulet Coulombe & Klieber (2025) 'An Adaptive Moving Average for Macroeconomic Monitoring', arXiv:2501.13222.

## Related ops

See also: `savitzky_golay_filter`, `hamilton_filter`, `hp_filter`, `ma_window` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
