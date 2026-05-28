# `harvey_newbold_encompassing` -- Harvey-Leybourne-Newbold (1998) forecast-encompassing test.

[Back to `equal_predictive_test` axis](../axes/equal_predictive_test.md) | [Back to L6](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `equal_predictive_test`, sub-layer `L6_A_equal_predictive`, layer `l6`.
> Standalone callable: `mf.functions.hn_test`.

## Function signature

```python
mf.functions.hn_test(
    e_a: np.ndarray,
    e_b: np.ndarray,
) -> HNTestResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `e_a` | `np.ndarray` | — | — | Forecast errors for model A (actual - forecast_a). |
| `e_b` | `np.ndarray` | — | — | Forecast errors for model B (actual - forecast_b). |

## Returns

`HNTestResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `stat` | `float or None` | HN test statistic |
| `pvalue` | `float or None` | One-sided p-value |
| `decision` | `bool` | Reject H0 at 5% |
| `n_obs` | `int` | Observations used |
| `encompassing` | `str` | Direction: a_over_b |

## Behavior

Tests the null that forecast f_1 encompasses f_2 -- i.e. the optimal linear combination of the two forecasts puts zero weight on f_2's error. Constructs ``d_t = e_a (e_a - e_b)`` from the per-period forecast errors and tests its mean against zero with a Newey-West HAC long-run variance and an HLN small-sample correction at horizon h>1. Asymmetric by construction (f_1 encompasses f_2 ≠ f_2 encompasses f_1).

**When to use**

Deciding whether one forecast contains all the information of another.

**When NOT to use**

Symmetric equal-MSE comparison -- use ``dm_diebold_mariano`` instead.

## In recipe context

Set ``params.equal_predictive_test = "harvey_newbold_encompassing"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L6 recipe fragment
params:
  equal_predictive_test: harvey_newbold_encompassing
```

## References

* macroforecast design Part 3, L6: 'tests must report (statistic, p-value, kernel, lag) and respect HAC dependence-correction.'
* Harvey, Leybourne & Newbold (1998) 'Tests for Forecast Encompassing', JBES 16(2): 254-259.

## Related ops

See also: `dm_diebold_mariano`, `gw_giacomini_white`, `multi` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
