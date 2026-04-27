# 4.1.3 failure_policy

- Parent: [4.1 Layer 0: Study Setup](index.md)
- Previous: [4.1.2 experiment_unit](experiment_unit.md)
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

| Choice | Status | What It Does | Current Runtime Behavior |
|---|---:|---|---|
| `fail_fast` | operational | Stop at the first failure. | Default. Direct recipe and sweep execution re-raise the first error. |
| `skip_failed_cell` | operational | Continue after a failed sweep cell or variant. | Sweep runner records failed variants in `study_manifest.json` and continues. Best for large controlled variation sweeps. |
| `skip_failed_model` | operational | Continue after a model-branch failure. | Direct recipe runtime can continue past target/model prediction-build failures and record failed components. |
| `retry_then_skip` | registry_only | Reserved policy: retry a failed unit, then skip if it still fails. | Not supported by the current runtime slice. Compiler reports not-supported if selected. |
| `fallback_to_default_hp` | registry_only | Reserved policy: if tuning fails, fall back to default hyperparameters. | Not supported by the current runtime slice. |
| `save_partial_results` | operational | Preserve completed artifacts even when later components fail. | Direct recipe and sweep paths treat it as a continue-on-failure policy where supported and record failure metadata. |
| `warn_only` | operational | Continue and emit warnings for recoverable failures. | Direct recipe and sweep paths emit `RuntimeWarning` while recording failed units. |

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
      research_design: controlled_variation
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
