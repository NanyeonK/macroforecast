# `clark_west` -- Clark-West (2007) MSE-adjusted nested-model predictive ability test.

[Back to `nested_test` axis](../axes/nested_test.md) | [Back to L6](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `nested_test`, sub-layer `L6_B_nested`, layer `l6`.
> Standalone callable: `mf.functions.cw_test`.

## Function signature

```python
mf.functions.cw_test(
    loss_small: np.ndarray,
    loss_large: np.ndarray,
    f_small: np.ndarray,
    f_large: np.ndarray,
) -> CWTestResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `loss_small` | `np.ndarray` | — | — | Squared losses for the small (restricted) model. |
| `loss_large` | `np.ndarray` | — | — | Squared losses for the large (unrestricted) model. |
| `f_small` | `np.ndarray` | — | — | Point forecasts for the small model. |
| `f_large` | `np.ndarray` | — | — | Point forecasts for the large model. |

## Returns

`CWTestResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `stat` | `float or None` | CW test statistic |
| `pvalue` | `float or None` | One-sided p-value |
| `decision` | `bool` | Reject H0 at 5% |
| `n_obs` | `int` | Observations used |
| `cw_adjustment` | `bool` | CW penalty applied |

## Behavior

Tests whether the large (unrestricted) model significantly outperforms the small (restricted, nested) model. Constructs the CW-adjusted statistic ``f_t = (loss_small - loss_large) + (f_small - f_large)^2``, removing the negative expected value bias that standard DM has in nested comparisons. One-sided test (H_a: large model improves on small); ``hln=False``.

**When to use**

Testing whether a larger model with additional regressors beats the restricted benchmark.

**When NOT to use**

Non-nested model comparisons -- use DM / GW (L6.A) instead. Forecast combination (use HN encompassing instead).

## In recipe context

Set ``params.nested_test = "clark_west"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L6 recipe fragment
params:
  nested_test: clark_west
```

## References

* macroforecast design Part 3, L6: 'tests must report (statistic, p-value, kernel, lag) and respect HAC dependence-correction.'
* Clark & West (2007) 'Approximately Normal Tests for Equal Predictive Accuracy in Nested Models', JoE 138(2): 291-311.

## Related ops

See also: `enc_new`, `enc_t`, `multi` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
