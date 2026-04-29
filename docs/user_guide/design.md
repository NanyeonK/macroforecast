# Design (Stage 0)

Stage 0 decides **the execution grammar**: which unit of work is run or compared, how axes sweep, how failures are handled, how deterministic the run is, and how work is parallelised. The simple default path uses the small operational subset needed for a one-cell comparison or a controlled model comparison.

**At a glance (defaults):**
- `study_scope = one_target_one_method` — one target, one forecasting path, one `comparison_sweep` cell.
- `axis_type = fixed` per axis — set to `sweep` only on the axis you are varying.
- `failure_policy = fail_fast` — stop on the first error.
- `reproducibility_mode = seeded_reproducible` — Python + numpy seeded; torch optional.
- `compute_mode = serial` — no parallelism.

Deviate when you explicitly want multiple targets, method comparisons, looser error handling, stricter determinism, or compute speedups.

---

## 0.1 `study_scope`

**Selection question**: How many targets and methods should the study compare?

**Default**: `one_target_one_method` for one fixed single-target path.

A one-path forecast is the one-cell case of `comparison_sweep`. A controlled model, feature, preprocessing, or representation comparison is the multi-cell case of the same grammar.

### Value catalog

| Value | Status | When to use | Verify |
|---|---|---|---|
| `one_target_one_method` | operational | One target, one fixed method path. | `tree_context["route_contract"] == "single_cell_executable"` |
| `one_target_compare_methods` | operational | One target with one or more supported method sweeps. | `tree_context["route_contract"] == "sweep_runner_executable"` when sweeps are present. |
| `multiple_targets_one_method` | operational | Multiple targets with one fixed method path. | `target_structure == "multi_target"`; one shared multi-target run. |
| `multiple_targets_compare_methods` | operational | Multiple targets with one or more supported method sweeps. | `target_structure == "multi_target"`; sweep runner parent when sweeps are present. |

### Compatibility guards

- `one_target_*` scopes require `target_structure = single_target`.
- `multiple_targets_*` scopes require `target_structure = multi_target`.
- Replication Library entries are normal YAML recipes; replication is not a Layer 0 Study Scope branch.

### Functions & features

- Auto-derivation: `macrocast.registry.stage0.study_scope.derive_study_scope_default`.
- Runners: `execute_recipe` for one-cell scopes, `compile_sweep_plan` / `execute_sweep` when downstream axes are swept.

### Recipe usage

```yaml
# Controlled model comparison: variants are executed by execute_sweep().
path:
  0_meta:
    fixed_axes:
      study_scope: one_target_compare_methods
  3_training:
    sweep_axes:
      model_family: [ar, ridge, lasso]
```

---

## 0.2 `axis_type`

**Selection question**: For a given axis, is it held fixed, swept, derived from other axes, or chosen by a rule?

**Default**: `fixed` — you set the axis to one value and it stays there.

Applies **per axis**, not per recipe. The compiler derives this from how the axis appears in `path["<layer>"]` (`fixed_axes` vs. `sweep_axes` vs. `conditional_axes`), so you almost never set it directly.

### Value catalog

| Value | Status | When to use | Verify |
|---|---|---|---|
| `fixed` | operational (default) | Default. Axis takes one value for the whole study. | Axis appears under `fixed_axes` in the layer spec. |
| `sweep` | operational | Axis varies across the sweep plan. | Axis appears under `sweep_axes`; each variant in `study_manifest.variants[*]`. |
| `nested_sweep` | operational | Two axes sweep together with a nested dependency. | Nested list structure in `sweep_axes`; variants enumerate the outer×inner product. |
| `conditional` | operational | Axis value that depends on another axis resolution. | Axis appears under `conditional_axes`; value is tracked in the tree_context for provenance (no runtime rule engine in v1.0). |
| `derived` | operational | Axis value inferred from the compiler (never user-supplied). | Manifest shows the derived value; not present in input YAML. |

### Functions & features

- Discovery: `macrocast.compiler.build._build_axis_selections` reads the three layer-spec forms and tags each axis with its `selection_mode`.
- Derivation: `macrocast.compiler.build._resolve_derived_axes` handles `derived`; rule registry for `conditional`.

### Recipe usage

```yaml
# nested_sweep: sweep model_family first; for quantile_linear, also sweep forecast_object.
path:
  3_training:
    sweep_axes:
      model_family: [ar, ridge, quantile_linear]
    conditional_axes:
      forecast_object:
        rule: quantile_model_requires_point_median
```

---

## 0.3 `failure_policy`

**Selection question**: What happens when a variant / cell fails at runtime?

**Default**: `fail_fast` — the first failure aborts the study so you can investigate. You can leave this axis out of a recipe unless you want a more tolerant run.

Loosen when you're running a large sweep and want a partial report rather than an abort.

### Value catalog

| Value | Status | When to use | Verify |
|---|---|---|---|
| `fail_fast` | operational (default) | Default. Any single-recipe run; sweeps during recipe development. | No change — run aborts on first error. |
| `skip_failed_cell` | operational | Large sweeps where the failure of one variant shouldn't stop the others. | `study_manifest["summary"]["skipped"]` and per-variant `compiler_status` / `compiler_blocked_reasons` record compile-invalid cells. |
| `skip_failed_model` | operational | Same pattern scoped to model families. | Failed variants remain in `study_manifest["sweep_plan"]["variants"]`. |
| `save_partial_results` | operational | Flush artefacts before aborting — useful when you want what completed so far. | Partial `run/` directories persist. |
| `warn_only` | operational | Never stop; emit a `RuntimeWarning` per failure. | stderr warnings plus per-variant status in `study_manifest`. |

### Functions & features

- Dispatch: `macrocast.execution.sweep_runner._extract_parent_failure_policy`, `_CONTINUE_ON_VARIANT_FAILURE` frozenset.
- Ablation: `macrocast.studies.ablation._ensure_baseline_failure_policy` auto-upgrades the baseline to `skip_failed_cell` unless the user sets something stricter.

### Recipe usage

```yaml
# 100-variant sweep — skip any cell that crashes instead of aborting.
path:
  0_meta:
    fixed_axes:
      failure_policy: skip_failed_cell
  3_training:
    sweep_axes:
      model_family: [ar, ridge, lasso, ...]
```

---

## 0.4 `reproducibility_mode`

**Selection question**: How hard do you want to pin the stochastic components?

**Default**: `seeded_reproducible` with seed `42` — Python `random`, numpy, and torch (if available) are seeded; no cudnn / BLAS determinism. You can leave this axis out of a recipe unless you want stricter or looser behavior.

Escalate when you need bit-identical reruns (paper replication) or when you don't care at all (exploratory drafting).

### Value catalog

| Value | Status | When to use | Verify |
|---|---|---|---|
| `seeded_reproducible` | operational (default) | You want the same result across reruns on the same machine without strict deterministic-library flags. | `manifest["reproducibility_applied"]["mode"] == "seeded_reproducible"`. |
| `best_effort` | operational | Same seed application as `seeded_reproducible`, but labeled as non-strict for CI/regression interpretation. | `manifest["reproducibility_applied"]["mode"] == "best_effort"`. |
| `strict_reproducible` | operational | Paper replication — bit-identical across machines. | `torch.use_deterministic_algorithms(True)`; `CUBLAS_WORKSPACE_CONFIG=:4096:8`; `RuntimeWarning` if `PYTHONHASHSEED` is unset. |
| `exploratory` | operational | Research drafting — don't seed at all. | manifest records `exploratory`; no seeds applied. |

### Functions & features

- Helper: `macrocast.execution.seed_policy.apply_reproducibility_mode(mode, seed, configure_torch=True)`.
- Called by `execute_recipe` before any stochastic step; result stored in `manifest["reproducibility_applied"]`.

### Recipe usage

```yaml
# Paper replication — strict determinism.
path:
  0_meta:
    fixed_axes:
      reproducibility_mode: strict_reproducible
    leaf_config:
      random_seed: 42
```

---

## 0.5 `compute_mode`

**Selection question**: How should execution work be laid out?

**Default**: `serial` — one variant / horizon / target / origin at a time. You can leave this axis out of a recipe unless the run has a natural parallel work unit.

Pick a parallel mode when wall-clock matters and the chosen level actually has multiple units of work. Navigator disables scope-incompatible choices: `parallel_by_model` requires a Study Scope that compares methods, and `parallel_by_target` requires a multiple-target Study Scope.

### Value catalog

| Value | Status | When to use | Verify |
|---|---|---|---|
| `serial` | operational (default) | Default. Any run where wall-clock isn't the bottleneck. | No parallelism; straight `for`-loop. |
| `parallel_by_model` | operational | Method-comparison Study Scope with many model_family variants. | `ThreadPoolExecutor(max_workers=min(n_variants, 4))` in `sweep_runner._run_variant`. |
| `parallel_by_horizon` | operational | Recipe with multiple horizons per target. | Pool inside `_rows_for_horizon` when `len(horizons) > 1`. |
| `parallel_by_target` | operational | Multiple-target Study Scope. | Pool inside `execute_recipe` when `len(targets) > 1`. |
| `parallel_by_oos_date` | operational | Long OOS windows — origin-level parallelism. | Pool inside `_rows_for_horizon` stage-2 when `len(origin_plan) > 1`. |

A parallel mode that doesn't have multiple units of work (e.g. `parallel_by_horizon` with a single-horizon recipe) is a silent no-op — not an error.

### Functions & features

- `macrocast.execution.sweep_runner.execute_sweep._extract_parent_compute_mode`.
- `macrocast.execution.build._build_predictions` handles `parallel_by_horizon` / `parallel_by_oos_date`.
- `macrocast.execution.build.execute_recipe` handles `parallel_by_target`.

### Recipe usage

```yaml
# 30-variant model sweep on a multi-core box.
path:
  0_meta:
    fixed_axes:
      compute_mode: parallel_by_model
  3_training:
    sweep_axes:
      model_family: [ar, ridge, lasso, elasticnet, ..., xgboost, lightgbm, mlp]
```

---

## Design (Stage 0) takeaways

- Four user-facing axes plus internal YAML grammar keep the operational surface narrow. Researchers who do not need failure-policy changes, stricter reproducibility, or parallelism write Layer 0 mostly by omission.
- Runner dispatch flows from `study_scope`. One fixed path and a controlled sweep both use the `comparison_sweep` route; the difference is whether any downstream `sweep_axes` are present.
- `failure_policy` + `reproducibility_mode` + `compute_mode` are three defaulted execution dials. Pick per-run, and let Navigator show which compute choices are disabled by the chosen Study Scope.
- Every resolved value lands in `manifest.json` — when a run does something unexpected, read the manifest before anything else.

Next: [Data (Stage 1)](data/index.md).
