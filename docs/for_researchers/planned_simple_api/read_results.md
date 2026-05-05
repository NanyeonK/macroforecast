# Read Results

> **API status note (current)**: this page uses the planned mf.forecast / mf.Experiment Python facade
> shape. Those are not yet exported from macroforecast.__all__. For working v0.6+ code, use
> macroforecast.run("recipe.yaml"), macroforecast.replicate("manifest.json"),
> the RecipeBuilder (macroforecast.scaffold.builder.RecipeBuilder), or
> python -m macroforecast scaffold. See [Simple Docs index](index.md) for the full status note.


`Experiment.run()` returns a facade over saved artifacts. Use the facade for normal analysis, and use artifact paths when exact files are needed.

Single run:

```python
import macroforecast as mf

result = mf.forecast(
    "fred_md",
    target="INDPRO",
    horizons=[1, 3],
    start="1980-01",
    end="2019-12",
)

forecasts = result.forecasts
metrics = result.metrics
comparison = result.comparison
manifest = result.manifest
```

Sweep:

```python
result = (
    mf.Experiment(
        dataset="fred_md",
        target="INDPRO",
        horizons=[1],
        start="1980-01",
        end="2019-12",
    )
    .compare_models(["ridge", "lasso"])
    .run()
)

ranking = result.compare("msfe")
forecasts = result.forecasts
variants = result.variants
manifest = result.manifest
```

Common attributes:

- `result.forecasts`: forecast rows from `predictions.csv`
- `result.predictions`: same table as `forecasts`
- `result.metrics`: one row per horizon, or per variant and horizon for sweeps
- `result.comparison`: compact comparison table
- `result.manifest`: provenance dictionary
- `result.artifact_dir`: comparison-cell artifact directory
- `result.output_root`: sweep artifact root

Single-run raw artifact access:

```python
result.metrics_json
result.comparison_json
result.file_path("predictions.csv")
result.read_json("manifest.json")
```

Sweep variant access:

```python
table = result.variants
best = result.compare("msfe").iloc[0]
variant = result.variant(best["variant_id"])
variant.forecasts
variant.manifest
```

Design rule: every number shown to a user should be traceable to a recipe, run, and artifact.
