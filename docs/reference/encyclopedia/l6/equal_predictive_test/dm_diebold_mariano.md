# `dm_diebold_mariano` -- Diebold-Mariano (1995) equal-predictive-ability test with Newey-West HAC SE.

[Back to `equal_predictive_test` axis](../axes/equal_predictive_test.md) | [Back to L6](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `equal_predictive_test`, sub-layer `L6_A_equal_predictive`, layer `l6`.
> Standalone callable: `mf.functions.dm_test`.

## Function signature

```python
mf.functions.dm_test(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
) -> DMTestResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `loss_a` | `np.ndarray` | — | — | Per-period losses for model A (e.g. squared errors). |
| `loss_b` | `np.ndarray` | — | — | Per-period losses for model B. |

## Returns

`DMTestResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `stat` | `float or None` | DM test statistic |
| `pvalue` | `float or None` | Two-sided p-value |
| `decision` | `bool` | Reject H0 at 5% |
| `n_obs` | `int` | Observations used |
| `hln_correction` | `bool` | HLN correction applied |

## Behavior

Pairwise test of equal expected loss between two forecasts. Implements DM with HLN small-sample correction (Harvey-Leybourne-Newbold 1997) and a configurable HAC kernel (``newey_west`` default, ``andrews`` / ``parzen`` available). Two-sided alternative tests equality of MSE / MAE losses.

**When to use**

Pairwise comparison of two non-nested forecasts.

**When NOT to use**

Nested-model comparisons -- use Clark-West (L6.B) instead.

## In recipe context

Set ``params.equal_predictive_test = "dm_diebold_mariano"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L6 recipe fragment
params:
  equal_predictive_test: dm_diebold_mariano
```

## References

* macroforecast design Part 3, L6: 'tests must report (statistic, p-value, kernel, lag) and respect HAC dependence-correction.'
* Diebold & Mariano (1995) 'Comparing Predictive Accuracy', JBES 13(3): 253-263.
* Harvey, Leybourne & Newbold (1997) 'Testing the equality of prediction mean squared errors', IJF 13(2): 281-291.

## Related ops

See also: `gw_giacomini_white`, `dmp_multi_horizon`, `multi` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
