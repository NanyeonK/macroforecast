# Design (Stage 0)

Stage 0 decides **the shape of the study**: which runner fires, how axes sweep, how failures are handled, how deterministic the run is, and how work is parallelised. Six axes cover 38 allowed values; the simple default path uses only the small operational subset needed for a single run or model comparison. Every axis ships a default that matches the most common research flow — most researchers leave all six alone and never read this page.

**At a glance (defaults):**
- `research_design = single_forecast_run` — one recipe, one forecast.
- `experiment_unit` is derived from `target_structure` + sweep shape (auto-picked; never needs to be set).
- `axis_type = fixed` per axis — set to `sweep` only on the axis you are varying.
- `failure_policy = fail_fast` — stop on the first error.
- `reproducibility_mode = seeded_reproducible` — Python + numpy seeded; torch optional.
- `compute_mode = serial` — no parallelism.

Deviate when you explicitly want multiple variants, different runners, looser error handling, stricter determinism, or compute speedups.

---

## 0.1 `research_design`

**Selection question**: Is this a single forecast, a horse race across variants, a Phase 8 paper bundle, or a replication of a prior recipe?

**Default**: `single_forecast_run` — one recipe produces one forecast and one metrics file. Most empirical work starts here.

### Value catalog

| Value | Status | When to use | Verify |
|---|---|---|---|
| `single_forecast_run` | operational (default) | Default. Any study that compares the chosen model against a benchmark on one dataset/target combination. | `manifest["research_design"] == "single_forecast_run"` |
| `controlled_variation` | operational | Horse-race studies — one axis sweeps across values, everything else held fixed. | One `study_manifest.json` per sweep plan; one `run/` directory per variant. |
| `study_bundle` | operational as a routing shape | Use only with an experiment unit that has a concrete runner contract. Unsupported wrapper families compile as `not_supported`. | `execution_status` is `ready_for_wrapper_runner` for supported wrappers, otherwise `not_supported`. |
| `replication_recipe` | operational | Re-running an existing recipe and asserting byte-identical output. | `execute_replication()` returns `ReplicationResult`; identical forecast + manifest hashes. |

### Functions & features

- Runner dispatch: `macrocast.compiler.build` resolves `research_design` + `target_structure` → `experiment_unit` via `derive_experiment_unit_default`.
- Artifacts: `single_forecast_run` → `execute_recipe()`; `controlled_variation` → `execute_sweep()`; `replication_recipe` → `execute_replication()`.

### Recipe usage

```yaml
# Sweep across model_family while holding everything else fixed.
path:
  0_meta:
    fixed_axes:
      research_design: controlled_variation
  3_training:
    sweep_axes:
      model_family: [ar, ridge, lasso, randomforest]
```

---

## 0.2 `experiment_unit`

**Selection question**: Which runner owns this recipe?

**Default**: Auto-derived from `target_structure` and sweep shape — you never need to set this explicitly.

The compiler calls `derive_experiment_unit_default(research_design, target_structure, model_axis_mode, feature_axis_mode, wrapper_family)` and picks one of the operational values below. Set it explicitly only when your recipe must declare a specific runner (e.g. for replication provenance).

### Value catalog

| Value | Status | When to use | Verify |
|---|---|---|---|
| `single_target_single_generator` | operational (default for single-target, no sweeps) | Default auto-derivation. | `manifest["experiment_unit"] == "single_target_single_generator"` |
| `single_target_generator_grid` | operational | Auto-picked when `model_family` sweeps. | manifest records the derived value. |
| `single_target_full_sweep` | registry_only | Grammar retained for future wrapper/orchestrator work; not exposed as runnable. | `execution_status == "not_supported"`. |
| `multi_target_separate_runs` | operational | Multi-target recipe, each target runs independently via `execute_separate_runs`. | N `run/` directories; `separate_runs_manifest.json`. |
| `multi_target_shared_design` | operational (default for multi-target) | Multi-target with shared preprocessing / benchmarks (default auto-derivation). | Single run directory; `predictions.csv` contains all targets. |
| `replication_recipe` | operational | Auto-derived when `research_design=replication_recipe`. | `ReplicationResult` artefact. |
| `benchmark_suite` | registry_only | Reserved for a future PaperReadyBundle/runtime contract. | `execution_status == "not_supported"`. |
| `ablation_study` | registry_only | Standalone `AblationSpec` runner exists, but compiled-recipe wrapper handoff is not wired. | `execution_status == "not_supported"`. |

### Compatibility guards

- `experiment_unit.requires_multi_target == True` (e.g. `multi_target_separate_runs`, `multi_target_shared_design`) requires `target_structure = multi_target_point_forecast`; else `blocked_by_incompatibility`.
- `experiment_unit.requires_multi_target == False` with `target_structure = multi_target_point_forecast` is also blocked.

### Functions & features

- Auto-derivation: `macrocast.design.derive.derive_experiment_unit_default`.
- Runners: `execute_recipe`, `execute_sweep`, `execute_separate_runs` (`macrocast.studies.multi_target`), `execute_replication` (`macrocast.studies.replication`). `execute_ablation` is standalone until a compiled-recipe contract is added.

### Recipe usage

```yaml
# Controlled model comparison: variants are executed by execute_sweep().
path:
  0_meta:
    fixed_axes:
      research_design: controlled_variation
  3_training:
    sweep_axes:
      model_family: [ar, ridge, lasso]
```

---

## 0.3 `axis_type`

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

## 0.4 `failure_policy`

**Selection question**: What happens when a variant / cell fails at runtime?

**Default**: `fail_fast` — the first failure aborts the study so you can investigate.

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

## 0.5 `reproducibility_mode`

**Selection question**: How hard do you want to pin the stochastic components?

**Default**: `seeded_reproducible` — Python `random`, numpy, and torch (if available) are seeded; no cudnn / BLAS determinism.

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

## 0.6 `compute_mode`

**Selection question**: Which level of the sweep should run in parallel?

**Default**: `serial` — one variant / horizon / origin at a time.

Pick a parallel mode when the sweep is big enough that wall-clock matters and the chosen level actually has multiple units of work.

### Value catalog

| Value | Status | When to use | Verify |
|---|---|---|---|
| `serial` | operational (default) | Default. Any run where wall-clock isn't the bottleneck. | No parallelism; straight `for`-loop. |
| `parallel_by_model` | operational | Sweep with many model_family variants. | `ThreadPoolExecutor(max_workers=min(n_variants, 4))` in `sweep_runner._run_variant`. |
| `parallel_by_horizon` | operational | Recipe with multiple horizons per target. | Pool inside `_rows_for_horizon` when `len(horizons) > 1`. |
| `parallel_by_target` | operational | Multi-target recipe, N targets. | Pool inside `execute_recipe` when `len(targets) > 1`. |
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

- Six axes, 38 allowed values, and the simple default path keeps the operational surface narrow. Researchers who don't need sweeps, parallelism, or strict reproducibility write Stage 0 by omission.
- Runner dispatch flows `research_design` → `experiment_unit`. The second is auto-derived; the first is the user-facing lever for "what kind of study is this?"
- `failure_policy` + `reproducibility_mode` + `compute_mode` are three independent dials. Pick per-run, don't carry them over from copy-pasted templates.
- Every resolved value lands in `manifest.json` — when a run does something unexpected, read the manifest before anything else.

Next: [Data (Stage 1)](data/index.md).
