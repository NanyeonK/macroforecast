# macrocast_start

`macrocast_start()` is a staged entry helper for testing and inspection.

Purpose:
- let the user select package stages explicitly
- inspect tree-path state without manually wiring internal modules
- provide a simple start surface for testing the package step-by-step

Available stages:
- `axes`
- `registries`
- `compile`
- `tree_context`
- `runs_preview`
- `manifest_preview`

Example:

```python
from macrocast import macrocast_start

out = macrocast_start(
    recipe_path='baselines/minimal_fred_md.yaml',
    stages=['compile', 'tree_context', 'manifest_preview'],
)
```
