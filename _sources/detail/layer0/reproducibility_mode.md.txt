# 4.0.3 Reproducibility

- Parent: [4.0 Layer 0: Study Scope](index.md)
- Previous: [4.0.2 Failure Handling](failure_policy.md)
- Current: `reproducibility_mode`
- Next: [4.0.4 Compute Layout](compute_mode.md)

`reproducibility_mode` controls seed resolution and deterministic runtime behavior.

It is applied before model execution. It affects Python, NumPy, optional torch, cuDNN, CUDA deterministic settings, and per-variant seed derivation.

## Where It Lives In Code

| Purpose | Function or object |
|---|---|
| Registry entries | `macrocast.registry.stage0.reproducibility_mode.REPRODUCIBILITY_MODE_ENTRIES` |
| Compiler seed requirement | `macrocast.compiler.build.compile_recipe_dict` |
| Manifest payload | `macrocast.compiler.build.compiled_spec_to_dict` |
| Execution payload reader | `macrocast.execution.build._reproducibility_spec` |
| Runtime context | `macrocast.execution.seed_policy.ReproducibilityContext` |
| Per-variant seed | `macrocast.execution.seed_policy.resolve_seed` |
| Current model seed helper | `macrocast.execution.seed_policy.current_seed` |
| Apply global RNG / backend flags | `macrocast.execution.seed_policy.apply_reproducibility_mode` |

## Choices

Read this axis as the seed and determinism policy. The stricter modes require `leaf_config.random_seed`; `exploratory` intentionally waives reproducibility.

### Quick Map

| Choice | Current State | Determinism Level |
|---|---|---|
| `strict_reproducible` | runnable | strongest |
| `seeded_reproducible` | runnable | normal seeded |
| `best_effort` | runnable | seeded, non-strict label |
| `exploratory` | runnable | intentionally non-deterministic |

### `strict_reproducible`

Use this for replication and CI-sensitive runs.

```yaml
path:
  0_meta:
    fixed_axes:
      reproducibility_mode: strict_reproducible
    leaf_config:
      random_seed: 42
```

Runtime behavior:

```text
seed source = deterministic per variant
seed key    = recipe_id | variant_id | model_family
backend     = torch/cuDNN deterministic flags where available
```

This mode also warns if `PYTHONHASHSEED` was not set before Python started.

### `seeded_reproducible`

Use this as the normal research default.

```yaml
path:
  0_meta:
    fixed_axes:
      reproducibility_mode: seeded_reproducible
    leaf_config:
      random_seed: 42
```

Runtime behavior:

```text
seed source = recipe base seed
backend     = Python / NumPy / torch seeds set
strict flags = not forced
```

This is reproducible enough for ordinary package runs, but it does not claim bit-identical backend behavior across all library/hardware combinations.

### `best_effort`

Use this when you want seeded behavior but do not want the run classified as strict.

```yaml
path:
  0_meta:
    fixed_axes:
      reproducibility_mode: best_effort
    leaf_config:
      random_seed: 42
```

Runtime behavior:

```text
install-time behavior = same as seeded_reproducible
semantic label        = non-strict
```

This is useful when deterministic flags are too costly or not meaningful for the backend.

### `exploratory`

Use this only for ad-hoc exploration where results should not be treated as reproducible evidence.

```yaml
path:
  0_meta:
    fixed_axes:
      reproducibility_mode: exploratory
```

Runtime behavior:

```text
global RNG reset = no
current_seed()   = fresh seed from NumPy's current RNG state
random_seed      = not required
```

## Compiler Contract

For Full recipes, `strict_reproducible` and `seeded_reproducible` require `leaf_config.random_seed`.

```yaml
path:
  0_meta:
    fixed_axes:
      reproducibility_mode: seeded_reproducible
    leaf_config:
      random_seed: 42
```

The compiler writes:

```json
"reproducibility_spec": {
  "reproducibility_mode": "seeded_reproducible",
  "random_seed": 42
}
```

The runtime then calls `apply_reproducibility_mode()` at the start of `execute_recipe()`.

## Strict Mode Details

`strict_reproducible` does more than set a seed:

- calls Python and NumPy seed setters;
- calls torch seed setters when torch is importable;
- sets `torch.backends.cudnn.deterministic=True`;
- sets `torch.backends.cudnn.benchmark=False`;
- calls `torch.use_deterministic_algorithms(True, warn_only=True)` where available;
- sets `CUBLAS_WORKSPACE_CONFIG=':4096:8'` if the environment did not already set it;
- warns if `PYTHONHASHSEED` is missing, because Python hash ordering must be fixed before interpreter startup.

## YAML Examples

Replication-grade:

```yaml
path:
  0_meta:
    fixed_axes:
      reproducibility_mode: strict_reproducible
    leaf_config:
      random_seed: 42
```

Default research run:

```yaml
path:
  0_meta:
    fixed_axes:
      reproducibility_mode: seeded_reproducible
    leaf_config:
      random_seed: 42
```

Exploration:

```yaml
path:
  0_meta:
    fixed_axes:
      reproducibility_mode: exploratory
```

## Guidance

Use `strict_reproducible` for replication and CI-sensitive studies.

Use `seeded_reproducible` for normal research runs.

Use `best_effort` when deterministic flags are too costly or not meaningful for the backend, but you still want seeded behavior.

Use `exploratory` only for ad-hoc work whose results should not be treated as reproducible evidence.
