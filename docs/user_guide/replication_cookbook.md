# Replication cookbook

Phase 6 ships `execute_replication` so a prior run (including an entire
sweep's parent recipe) can be re-executed under a set of overrides, with a
machine-readable diff report that captures what changed.

```python
from macrocast import execute_replication, apply_overrides
```

## Minimum viable example

```python
import yaml
from macrocast.compiler.build import compile_recipe_dict
from macrocast.execution.build import execute_recipe
from macrocast import execute_replication

recipe = yaml.safe_load(open("examples/recipes/replication-synthetic.yaml"))
compiled = compile_recipe_dict(recipe).compiled
src = execute_recipe(
    recipe=compiled.recipe_spec,
    preprocess=compiled.preprocess_contract,
    output_root="out/src-run",
    local_raw_source="tests/fixtures/fred_md_ar_sample.csv",
    provenance_payload={"compiler": {"reproducibility_spec": {
        "reproducibility_mode": "seeded_reproducible", "seed": 42,
    }}},
)

result = execute_replication(
    source_recipe_dict=recipe,
    overrides={"path.3_training.fixed_axes.model_family": "lasso"},
    source_artifact_dir=src.artifact_dir,
    output_root="out/replay-run",
    local_raw_source="tests/fixtures/fred_md_ar_sample.csv",
    provenance_payload={"compiler": {"reproducibility_spec": {
        "reproducibility_mode": "seeded_reproducible", "seed": 42,
    }}},
)
print(result.overrides_applied)
print(result.byte_identical_predictions)
print(result.diff_report_path)
```

## Override path convention

Literal dotted keys into the recipe dict — see `ablation_cookbook.md` or the module docstring at `macrocast/compiler/override_diff.py`.

## Byte-identical round-trip

Calling `execute_replication` with `overrides={}` over the *same* recipe and the same reproducibility settings reproduces the source run's `predictions.csv` byte-for-byte. This is the v1.0 synthetic replication gate; see `docs/examples/synthetic_replication_roundtrip.md`.

## Report schema

`replication_diff.json` fields:

- `schema_version` — `"1.0"`
- `source_recipe_id`, `replayed_recipe_id`
- `overrides_applied` — caller-supplied overrides dict
- `override_diff_entries[]` — same shape as `apply_overrides` return: `{path, old, new}`
- `source_artifact_dir`, `replayed_artifact_dir`
- `source_package_version`, `replayed_package_version` — lifted from each run's `manifest.json` when available
- `metrics_delta` — dict keyed by metric name, values `{source, replayed, delta_abs, delta_pct}`; `delta_pct` is `None` for near-zero sources
- `byte_identical_predictions` — `True` only when `overrides` is empty and the `predictions.csv` SHA-256 hashes match
- `created_at_utc`
