# Path Resolver

The path resolver compiles a YAML recipe and reports whether the selected path can execute.

Use it after inspecting the tree and before running a recipe.

## CLI

```bash
macroforecast-navigate resolve examples/recipes/model-benchmark.yaml
```

## Output

The resolver returns:

| Field | Meaning |
|---|---|
| `input_yaml_path` | Recipe path passed to the resolver. |
| `execution_status` | Compiler route status. |
| `warnings` | Non-fatal warnings. |
| `blocked_reasons` | Reasons a path cannot execute. |
| `tree_context` | Canonical route context, fixed axes, sweep axes, leaf config, and route owner. |
| `layer3_capability_matrix` | Active forecast-generation cell and gated future cells. |

## Status Meanings

| Status | Meaning |
|---|---|
| `executable` | The recipe can run with the current runtime. |
| `blocked_by_incompatibility` | Values are valid individually but cannot compose in the current runtime. |
| `compile_error` | The YAML contains invalid axis values, missing required fields, or governance violations. |
| `ready_for_wrapper_runner` | The path is valid but belongs to a higher-level wrapper such as a replication or study runner. |

## Canonical Path Discipline

The resolver is the last authority before execution. The Tree Navigator can explain disabled branches, but the compiler still decides whether the complete route is executable.

Examples:

- `tcode_policy=official_tcode_only` with `scaling_policy=standard` is invalid because `official_tcode_only` cannot carry extra preprocessing.
- Coulombe-style `t-code + standardize` must use `tcode_policy=official_tcode_then_extra_preprocess`, `preprocess_order=official_tcode_then_extra`, and a train-only fit scope.
- Raw-panel iterated forecasting requires an explicit future-X path assumption.

## Python

```python
from macroforecast.navigator.core import resolve_yaml_path

resolved = resolve_yaml_path("examples/recipes/model-benchmark.yaml")
print(resolved["execution_status"])
```
