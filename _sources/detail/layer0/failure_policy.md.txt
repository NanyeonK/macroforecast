# 4.1.3 failure_policy

- Parent: [4.1 Layer 0: Study Setup](index.md)
- Previous: [4.1.2 study_scope](study_scope.md)
- Current: `failure_policy`
- Next: [4.1.4 reproducibility_mode](reproducibility_mode.md)

`failure_policy` controls what happens when a recipe, sweep variant, model branch, target branch, or cell fails.

This is a runtime discipline axis. It does not change the statistical model. It changes whether the runner stops, skips, warns, or preserves partial artifacts.

## Where It Lives In Code

| Purpose | Function or object |
|---|---|
| Registry entries | `macrocast.registry.stage0.failure_policy.FAILURE_POLICY_ENTRIES` |
| Compiler runtime support gate | `macrocast.compiler.build._execution_status` |
| Manifest payload | `macrocast.compiler.build.compiled_spec_to_dict` |
| Execution payload reader | `macrocast.execution.build._failure_policy_spec` |
| Direct recipe runtime | `macrocast.execution.build.execute_recipe` |
| Sweep parent policy reader | `macrocast.execution.sweep_runner._extract_parent_failure_policy` |
| Sweep runtime | `macrocast.execution.sweep_runner.execute_sweep` |
| Ablation default helper | `macrocast.studies.ablation._ensure_baseline_failure_policy` |

## Choices

Read this axis as the run's error-handling policy. It does not change the model; it changes whether the runtime stops or records failed units and continues.

### Quick Map

| Choice | Current State | Best Use |
|---|---|---|
| `fail_fast` | runnable | debugging, replication |
| `skip_failed_cell` | runnable | large sweeps |
| `skip_failed_model` | runnable | multi-model direct runs |
| `save_partial_results` | runnable | long runs where partial artifacts matter |
| `warn_only` | runnable | exploratory runs |
| `retry_then_skip` | reserved | planned retry policy |
| `fallback_to_default_hp` | reserved | planned tuning fallback |

### `fail_fast`

Use this when a failure should stop the run immediately.

```yaml
path:
  0_meta:
    fixed_axes:
      failure_policy: fail_fast
```

Runtime behavior:

```text
direct recipe = re-raise first error
sweep runner  = stop at first failed variant
```

### `skip_failed_cell`

Use this for large controlled sweeps where some cells may be invalid.

```yaml
path:
  0_meta:
    fixed_axes:
      study_scope: one_target_compare_methods
      failure_policy: skip_failed_cell
```

Runtime behavior:

```text
sweep runner = record failed variant in study_manifest.json
next step    = continue with remaining variants
```

### `skip_failed_model`

Use this when one model branch may fail but other target/model branches should still run.

```yaml
path:
  0_meta:
    fixed_axes:
      failure_policy: skip_failed_model
```

Runtime behavior:

```text
direct recipe = continue past recoverable model/prediction failures
artifact      = record failed components
```

### `save_partial_results`

Use this when completed artifacts are valuable even if a later component fails.

```yaml
path:
  0_meta:
    fixed_axes:
      failure_policy: save_partial_results
```

Runtime behavior:

```text
direct recipe = preserve completed outputs where supported
sweep runner  = preserve completed variants and failure metadata
```

### `warn_only`

Use this for exploratory runs where recoverable failures should be visible but not fatal.

```yaml
path:
  0_meta:
    fixed_axes:
      failure_policy: warn_only
```

Runtime behavior:

```text
warning type = RuntimeWarning
artifact     = failed units recorded
run status   = continues where recoverable
```

### `retry_then_skip`

This is reserved policy grammar. It describes retrying a failed unit and skipping it if retry also fails.

```yaml
path:
  0_meta:
    fixed_axes:
      failure_policy: retry_then_skip
```

Current status:

```text
status = registry_only
compiler = reports not-supported for runnable recipes
```

### `fallback_to_default_hp`

This is reserved policy grammar for tuning failures.

```yaml
path:
  0_meta:
    fixed_axes:
      failure_policy: fallback_to_default_hp
```

Current status:

```text
status = registry_only
runtime = no current fallback executor
```

## Failure Scope

The same axis is read at two levels:

- Sweep level: `execute_sweep()` reads it from the parent recipe with `_extract_parent_failure_policy()`.
- Direct recipe level: `execute_recipe()` reads it from compiler provenance with `_failure_policy_spec()`.

This matters because `skip_failed_cell` is mainly a sweep-cell policy, while `skip_failed_model`, `save_partial_results`, and `warn_only` are also meaningful inside direct recipe execution.

## YAML

```yaml
path:
  0_meta:
    fixed_axes:
      failure_policy: fail_fast
```

For a controlled sweep:

```yaml
path:
  0_meta:
    fixed_axes:
      study_scope: one_target_compare_methods
      failure_policy: skip_failed_cell
```

For exploratory runs where recoverable failures should be visible but not fatal:

```yaml
path:
  0_meta:
    fixed_axes:
      failure_policy: warn_only
```

## Runtime Artifacts

The compiler writes:

```json
"failure_policy_spec": {
  "failure_policy": "fail_fast"
}
```

Sweep execution writes variant status and error text into `study_manifest.json`. Direct recipe execution records failed components and can still write partial outputs when the selected policy allows continuation.

## Guidance

Use `fail_fast` for debugging and replication.

Use `skip_failed_cell` for large sweeps where invalid combinations are expected.

Use `warn_only` for exploratory work where warnings should be visible in logs but the run should continue.

Use registry-only policies only when documenting planned contracts, not for runnable recipes.
