# Ablation cookbook

An *ablation study* answers the question *"how much does each component of
this pipeline matter?"* by holding a baseline recipe fixed and toggling one
piece at a time. Phase 6 ships the runner:

```python
from macrocast import AblationSpec, execute_ablation
```

`execute_ablation` is a thin layer over the Phase 1 sweep runner —
`baseline + N drop-one variants` are assembled into a `SweepPlan` and
executed via `execute_sweep`, then per-component metric deltas are written
to `<output_root>/ablation_report.json`.

## Minimum viable example

```python
import yaml
from macrocast import AblationSpec, execute_ablation

baseline = yaml.safe_load(open("examples/recipes/ablation-preprocessing.yaml"))

spec = AblationSpec(
    baseline_recipe_dict=baseline,
    components_to_ablate=(
        ("path.3_training.fixed_axes.model_family", "lasso"),
        ("path.3_training.fixed_axes.benchmark_family", "historical_mean"),
    ),
)

result = execute_ablation(
    spec=spec,
    output_root="out/ablation-demo",
    local_raw_source="tests/fixtures/fred_md_ar_sample.csv",
)
print(result.size, "variants ran")          # 1 baseline + 2 ablations
print(result.manifest_path)                 # study_manifest.json
```

Afterwards `out/ablation-demo/ablation_report.json` contains the delta of
each ablated variant vs the baseline, keyed by the override path that was
neutralised.

## Choosing a *neutral value*

"Ablating" a component means replacing its axis value with an *off* /
trivial value. What "off" means depends on the axis:

| Axis (example override path) | Typical neutral value |
|---|---|
| `path.2_preprocessing.fixed_axes.scaling_policy` | `none` |
| `path.2_preprocessing.fixed_axes.dimensionality_reduction_policy` | `none` |
| `path.2_preprocessing.fixed_axes.feature_selection_policy` | `none` |
| `path.3_training.fixed_axes.model_family` | a simpler model (`ols`, `lasso`, …) |
| `path.3_training.fixed_axes.feature_builder` | `autoreg_lagged_target` (target lags only) |
| `path.3_training.fixed_axes.benchmark_family` | `historical_mean` |

The runner has no opinion on what "neutral" is — you choose the replacement
value per component and pass it in `components_to_ablate`.

## Override path convention

Paths are literal dotted keys into the recipe dict. Concretely,

```
path.<layer>.fixed_axes.<axis_name>
```

for layer-level axes, or the corresponding `leaf_config` path for things
like `leaf_config.target`. `apply_overrides` (used internally) will raise
`KeyError` if a path does not resolve, so typos surface immediately.

## Report schema

`ablation_report.json` fields:

- `schema_version` — currently `"1.0"`
- `ablation_study_id` — either user-supplied or auto-derived (`abl-<12-hex>`
  over baseline + components)
- `baseline_variant_id` — always `"v-baseline"`
- `baseline_metrics` — per-component metrics dict from the baseline run's
  `metrics_summary`
- `components[]` — one entry per ablation, each with `axis_name`,
  `original_value`, `neutral_value`, `variant_id`, `status`, `metrics`, and
  `delta_vs_baseline` (source/replayed/delta_abs/delta_pct per metric key)
- `package_version`, `created_at_utc`
