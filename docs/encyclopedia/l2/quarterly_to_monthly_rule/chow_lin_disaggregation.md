# `chow_lin_disaggregation` -- chow_lin_disaggregation op.

[Back to `quarterly_to_monthly_rule` axis](../axes/quarterly_to_monthly_rule.md) | [Back to L2](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `quarterly_to_monthly_rule`, sub-layer `l2_a`, layer `l2`.

## In recipe context

Set ``params.quarterly_to_monthly_rule = "chow_lin_disaggregation"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L2 recipe fragment
params:
  quarterly_to_monthly_rule: chow_lin_disaggregation
```

## References

* macroforecast design, L2: see design docs for chow_lin_disaggregation.

## Related ops

See also: `step_backward` (on the same axis).

_Last reviewed 2026-05-22 by macroforecast author._
