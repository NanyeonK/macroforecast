# 4.1.4 reproducibility_mode

- Parent: [4.1 Layer 0: Study Setup](index.md)
- Previous: [4.1.3 failure_policy](failure_policy.md)
- Current: `reproducibility_mode`
- Next: [4.1.5 compute_mode](compute_mode.md)

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

| Choice | Status | What It Does | Seed Rule |
|---|---:|---|---|
| `strict_reproducible` | operational | Pin Python, NumPy, torch, cuDNN, torch deterministic algorithms, and CUBLAS workspace where possible. Warns if `PYTHONHASHSEED` is not set before Python starts. | `resolve_seed()` derives a deterministic per-variant seed from `(recipe_id, variant_id, model_family)`. |
| `seeded_reproducible` | operational | Pin Python, NumPy, and torch seeds, but do not force strict backend deterministic flags. | Uses one base seed for the run. |
| `best_effort` | operational | Same install-time behavior as `seeded_reproducible`, but explicitly marks the run as non-strict. | Uses one base seed for the run. |
| `exploratory` | operational | Do not reset global RNG state. Use this only when reproducibility is intentionally waived. | `current_seed()` receives a fresh non-deterministic seed from NumPy's current RNG state. |

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
