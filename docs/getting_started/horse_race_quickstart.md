# Horse-Race Quickstart

This quickstart shows the core macrocast v0.3 identity: take one recipe,
sweep one axis, and get a study-level bundle comparing variants.

## Install

```
pip install macrocast
```

## One recipe, four variants

Create `horse-race-model.yaml` (or copy from `examples/recipes/`):

```yaml
recipe_id: horse-race-model
path:
  0_meta:
    fixed_axes:
      research_design: controlled_variation
  3_training:
    fixed_axes:
      framework: rolling
      benchmark_family: zero_change
      feature_builder: raw_feature_panel
    sweep_axes:
      model_family: [ridge, lasso, elasticnet, bayesianridge]
  # ... plus fixed axes for 1_data_task, 2_preprocessing, 4..7
```

A `sweep_axes` entry on any layer turns that axis into a dimension the
runner will walk through. Axes in `fixed_axes` are held constant across
all variants, which is what makes the comparison fair.

## Compile and run

```python
from pathlib import Path
from macrocast import compile_recipe_yaml, compile_sweep_plan, execute_sweep

# compile_recipe_yaml returns the raw dict via load_recipe_yaml;
# compile_sweep_plan takes that dict and expands sweep_axes.
import yaml
recipe_dict = yaml.safe_load(Path("horse-race-model.yaml").read_text())

plan = compile_sweep_plan(recipe_dict)
print(f"study_id={plan.study_id}, size={plan.size}")

result = execute_sweep(
    plan=plan,
    output_root=Path("runs/horse-race-model"),
)
print(f"success={result.successful_count}, failed={result.failed_count}")
```

## What you get

```
runs/horse-race-model/
  study_manifest.json        # Schema v1 — one per study
  .raw_cache_shared/         # FRED cache shared across variants
  variants/
    v-<hash1>/
      manifest.json
      predictions.csv
      metrics.json
    v-<hash2>/
      ...
```

The `study_manifest.json` summarises every variant — its axis values,
status, artifact location, metrics, runtime, and the shared sweep plan.
It is the single entry point for downstream analysis (decomposition in
Phase 7, paper-ready bundle in Phase 8).

## Determinism guarantee

Each variant receives a stable, axis-value-derived seed via
`reproducibility_mode=strict_reproducible` (see the
[Reproducibility Policy](../dev/reproducibility_policy.md)). Running the
same plan twice produces byte-identical predictions per variant.

## Next

- [Sweep recipe grammar](../user_guide/sweep_recipes.md)
- [controlled_variation guide](../user_guide/controlled_variation.md)
- [execute_sweep API reference](../api/sweep_runner.md)
