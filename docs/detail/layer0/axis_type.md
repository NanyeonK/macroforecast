# Legacy Note: `axis_type`

`axis_type` is not a user-facing Layer 0 axis in the current layer-contract system.

Current behavior:

- list layers expose `fixed_axes` and non-sweepable evaluation/export axes;
- graph layers expose explicit DAG nodes and sinks;
- sweep behavior is inferred from recipe structure where supported;
- L0 axes are not sweepable.

Do not set `axis_type` directly in new recipes.

See [Layer 0](index.md) and [Layer Contract Design](../../user_guide/design.md).
