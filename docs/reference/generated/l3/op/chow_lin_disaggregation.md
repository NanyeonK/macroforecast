# `chow_lin_disaggregation` -- Chow-Lin (1971) regression-based temporal disaggregation from quarterly to monthly frequency.

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.chow_lin_disaggregation`.

## Function signature

```python
mf.functions.chow_lin_disaggregation(...)
```

## Behavior

Implements the best linear unbiased interpolation procedure of Chow & Lin (1971). A monthly indicator series (``params.chow_lin_indicator``) is regressed on the quarterly target observations using OLS; fitted values and residuals are distributed to monthly frequency. The quarterly sum-constraint is preserved: the sum of the three monthly disaggregated values equals the original quarterly value.

Algorithm steps:

1. Resample the monthly indicator to quarterly frequency by summing.
2. Fit OLS: quarterly_target ~ quarterly_indicator (with intercept).
3. Compute fitted quarterly values and residuals.
4. Spread residuals evenly across the three months of each quarter.
5. Reconstruct monthly series as ``fitted_monthly + residual_monthly``.

The monthly indicator column must be present in the input panel. If ``params.chow_lin_indicator`` is ``None``, the runtime selects the monthly column with highest absolute correlation to the quarterly observations automatically via ``_default_chow_lin_indicator``.

**When to use**

Quarterly-to-monthly temporal disaggregation when a correlated monthly indicator is available (e.g., industrial production as indicator for quarterly GDP).

**When NOT to use**

When no informative monthly indicator exists -- use step_backward (conservative) or linear_interpolation (smooth) instead. Also not appropriate when the series is already at monthly frequency.

## In recipe context

Set ``params.op = "chow_lin_disaggregation"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: chow_lin_disaggregation
```

## References

* Chow & Lin (1971) 'Best Linear Unbiased Interpolation, Distribution, and Extrapolation of Time Series by Related Series', Review of Economics and Statistics 53(4): 372-375. (doi:10.2307/1928739)

## Related ops

See also: `step_backward`, `linear_interpolation` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
