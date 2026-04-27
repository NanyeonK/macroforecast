# Layer 0 Appendix: axis_type

`axis_type` is registry grammar for how an axis is selected:

- fixed;
- sweep;
- nested sweep;
- conditional;
- derived.

Users normally do not set `axis_type` directly. They express it through the YAML section where an axis appears: `fixed_axes`, `sweep_axes`, or `conditional_axes`. Derived axes are resolved by the compiler.

Current detailed source: [Design Stage 0: axis_type](../../user_guide/design.md#axis-type).
