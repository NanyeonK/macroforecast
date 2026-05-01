# Legacy Note: `study_scope`

`study_scope` is not a current Layer 0 axis in the layer-contract system.

Current behavior:

- recipe shape, target count, and sweep shape determine the study scope;
- the derived value is recorded in manifest metadata;
- L0 itself exposes only `failure_policy`, `reproducibility_mode`, and `compute_mode`.

Do not write new layer-contract recipes that treat `study_scope` as a scientific or runtime axis.

See [Layer 0](index.md) and [Layer Contract Design](../../user_guide/design.md).
