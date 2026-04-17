# Infra: Study Manifest Schema

## 1. What

The `study_manifest.json` schema (v1) emitted by `SweepRunner` and consumed by
decomposition and bundle tooling. Single-path runs already write a per-run manifest;
the study manifest is the sweep-level superset that records the parent recipe, the
axes being swept, and the status of every variant. Downstream phases (decomposition
in phase-07, bundle export in phase-08) load this file to locate variant artifacts,
reconstruct which axis combinations were successful, and verify version/seed
compatibility before aggregation.

## 2. Used by phases

- phase-01 — SweepRunner produces the manifest at study creation and updates it as
  variants complete
- phase-07 — decomposition reads `variants[].artifact_dir` and `metrics_summary` to
  assemble the decomposition table
- phase-08 — bundle export includes the manifest verbatim and uses `variant_id` as
  the stable key for per-variant bundle subdirectories

## 3. API spec

Schema v1 (JSON):

```json
{
  "schema_version": "1.0",
  "study_id": "sha256-of-plan-canonical",
  "study_mode": "controlled_variation_study",
  "created_at_utc": "2026-04-17T10:30:00Z",
  "parent_recipe": { "...": "RecipeSpec (canonical form)" },
  "sweep_plan": {
    "axes_swept": ["model_family", "scaling_policy"],
    "variants": [
      {
        "variant_id": "v-a1b2c3d4",
        "axis_values": {"model_family": "ridge", "scaling_policy": "none"},
        "status": "success",
        "artifact_dir": "variants/v-a1b2c3d4/",
        "metrics_summary": {"msfe": 0.023, "relative_msfe": 0.87},
        "seed_used": 1234,
        "runtime_seconds": 45.2,
        "failure_log_ref": null
      }
    ],
    "successful_count": 10,
    "failed_count": 2
  },
  "tree_context": { "...": "existing tree context block" },
  "git_commit": "abc123",
  "package_version": "0.3.0"
}
```

`status` is one of `"success" | "failed" | "skipped"`. `failure_log_ref` is `null`
on success or a relative path like `"variants/v-a1b2c3d4/failures.json"` otherwise.

## 4. Implementation notes

Invariants enforced by the writer and checked by the consumer:

- `schema_version == "1.0"` exactly; readers reject other values and emit a clear
  message pointing at migration notes
- `study_id = sha256(canonical-JSON(sweep_plan))` **excluding** the mutable fields
  `variants[].status`, `variants[].artifact_dir`, `variants[].runtime_seconds`,
  `variants[].metrics_summary`, `variants[].failure_log_ref` — the ID must not
  change as variants complete
- `variant_id = "v-" + sha256(canonical-JSON(axis_values))[:8]` and must be stable
  across re-runs with the same axis values

Storage:

- Location: `<output_root>/study_manifest.json`
- Format: pretty-printed JSON (2-space indent), UTF-8, trailing newline
- Writes are atomic — write to `study_manifest.json.tmp` then `os.replace`

Canonical JSON means sorted keys, no trailing whitespace, UTF-8, `ensure_ascii=False`.

## 5. Test requirements

`tests/test_sweep_manifest_schema.py` covers:

- JSONSchema validation of a generated manifest against `schemas/study_manifest_v1.json`
- `study_id` is stable across two runs with the same plan but different outcomes
- `variant_id` is stable across re-runs for the same `axis_values`
- `successful_count + failed_count + skipped_count == len(variants)` at terminal state
- `schema_version` mismatch raises a specific error class on read
- Atomic write — simulating a crash mid-write leaves the old file intact

## 6. Owner / ADR references

Owned by phase-01 (producer) with read contract fixed for phase-07 and phase-08.
Related to ADR-001 (iteration sharing) — the manifest is produced by the shared
iteration layer that both single-path and sweep modes reuse. Schema is versioned so
future v2 migrations can be introduced non-breakingly through the `schema_version`
field.
