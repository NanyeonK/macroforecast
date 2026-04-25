# Decision Tree Navigator

The package should not force users to read the full API surface before they can
run a study. The navigator is the constraint-aware layer on top of the recipe
registry and compiler.

It answers four questions for every branch:

- what can be selected;
- what is disabled by the current path;
- why it is disabled;
- where the selected value lands in the canonical recipe path.

The implementation lives in `macrocast.navigator`. The CLI entry point is
`macrocast-navigate`.

## Tree Navigator

Inspect a recipe:

```bash
macrocast-navigate tree examples/recipes/model-benchmark.yaml
```

Show only the forecast-construction contract layers:

```bash
macrocast-navigate tree examples/recipes/model-benchmark.yaml --upstream-only
```

The JSON payload contains:

- `canonical_path`: the current Layer 0-7 path split into `fixed_axes`,
  `sweep_axes`, and `leaf_config`;
- `tree`: one entry per axis, with `options`, `status`, `enabled`,
  `disabled_reason`, and `canonical_path_effect`;
- `compatibility`: active constraint rules and downstream recommendations;
- `compile_preview`: the compiler route status and capability matrix.

Minimal Python use:

```python
from macrocast.navigator import build_navigation_view, load_recipe

recipe = load_recipe("examples/recipes/model-benchmark.yaml")
view = build_navigation_view(recipe)
```

## Compatibility Engine View

The first runtime-backed compatibility rules are intentionally explicit:

| Current selection | Effect |
|---|---|
| `importance_method=tree_shap` | `model_family` keeps only tree generators: `randomforest`, `extratrees`, `gbm`, `xgboost`, `lightgbm`, `catboost`. |
| `importance_method=linear_shap` | `model_family` keeps linear estimators. |
| `forecast_object=quantile` | `model_family=quantile_linear` is the current operational generator. Quantile-oriented downstream metrics/tests should be preferred where available. |
| `forecast_object=direction` | direction-family tests such as `pesaran_timmermann` and `binomial_hit` are recommended. |
| `forecast_object=interval` or `density` | density/interval calibration tests live on the `density_interval` axis. |
| `model_family in {lstm, gru, tcn}` | current runtime uses the univariate target-history sequence/autoreg path. Full multivariate `feature_runtime=sequence_tensor` remains gated. |
| `forecast_type=iterated` with raw-panel features | `leaf_config.exogenous_x_path_policy` selects `hold_last_observed`, `observed_future_x`, `scheduled_known_future_x`, or `recursive_x_model`/`ar1`. |

The navigator does not replace compiler validation. It makes the compiler's
constraint surface visible before users generate a YAML file.

Resolve a route:

```bash
macrocast-navigate resolve examples/recipes/model-benchmark.yaml
```

## Replication Library

List known replication routes:

```bash
macrocast-navigate replications
```

Write a runnable recipe:

```bash
macrocast-navigate replications goulet-coulombe-2021-fred-md-ridge \
  --write-yaml recipes/gc2021-ridge.yaml
```

Each replication entry contains:

- paper name;
- short description;
- exact tree path;
- recipe YAML;
- one-line CLI command;
- notebook snippet;
- expected outputs;
- deviations from the original paper.

Current built-in entries:

| ID | Paper / route | Purpose |
|---|---|---|
| `goulet-coulombe-2021-fred-md-ridge` | Goulet Coulombe et al. (2021), FRED-MD ridge path | Paper-style macro ML route with official transforms, raw-panel ridge, AR-BIC benchmark, and MSFE. |
| `synthetic-replication-roundtrip` | Synthetic fixture route | Small route for testing recipe lowering and artifact contracts. |

## Generate YAML, Then Run

CLI:

```bash
macrocast-navigate replications synthetic-replication-roundtrip \
  --write-yaml /tmp/synthetic-replication.yaml

macrocast-navigate run /tmp/synthetic-replication.yaml \
  --local-raw-source tests/fixtures/fred_md_ar_sample.csv \
  --output-root /tmp/macrocast-synthetic
```

Notebook:

```python
from macrocast.navigator import replication_recipe_yaml
from macrocast import compile_recipe_dict, run_compiled_recipe
import yaml

recipe = yaml.safe_load(replication_recipe_yaml("synthetic-replication-roundtrip"))
compiled = compile_recipe_dict(recipe)
result = run_compiled_recipe(
    compiled.compiled,
    output_root="/tmp/macrocast-synthetic",
    local_raw_source="tests/fixtures/fred_md_ar_sample.csv",
)
```

## Current Boundary

Layers 0-3 are upstream forecast-construction contracts. They are where the
navigator must be strict because bad choices can prevent forecast generation or
create leakage. Layers 4-7 are downstream evaluation, output, inference, and
interpretation consumers. They still have artifact contracts, but they consume
`predictions.csv`, `forecast_payloads.jsonl`, manifests, and error tables
rather than building the forecast representation.
