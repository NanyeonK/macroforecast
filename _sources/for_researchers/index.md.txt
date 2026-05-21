# For researchers

You want to run a macro-forecasting study, get the artifacts, and read the
results. You do **not** want to author new model classes or modify the package
source. Start here.

## If you want to ...

| Goal | Page |
|---|---|
| Set up the package and run your first recipe | [Quickstart](quickstart.md) |
| Build your first study with a working recipe template | [First study](first_study.md) |
| See what the runtime executes today vs. what is schema-only | [Runtime support matrix](../recipe_api/runtime_support.md) |
| Understand the output directory and manifest layout | [Understanding output](../recipe_api/output.md) |
| Look up FRED-MD / FRED-QD / FRED-SD column dictionaries | [FRED datasets](../recipe_api/fred_datasets.md) |
| Browse the recipe gallery (runnable examples) | [Recipe gallery](../recipe_api/gallery.md) |
| Use the high-level Python facade (`mf.forecast` / `mf.Experiment`) | [Simple API](simple_api/index.md) |
| Bring your own monthly or quarterly data (CSV / Parquet) | [Bring your own data](user_data_workflow.md) |

## Public API at a glance

```python
import macroforecast as mf

# Run any recipe end-to-end. Iterates every {sweep: [...]} cell, applies
# L0 failure_policy + seed, returns ManifestExecutionResult.
result = mf.run("recipe.yaml", output_directory="out/")

# Re-execute the stored manifest and verify per-cell sink hashes match
# bit-for-bit.
replication = mf.replicate("out/manifest.json")
assert replication.sink_hashes_match
```

For the full curated reference see [`encyclopedia/public_api.md`](../encyclopedia/public_api.md). Browse every recipe axis / option in the [encyclopedia](../encyclopedia/index.md).

```{toctree}
:hidden:
:maxdepth: 1

quickstart
first_study
user_data_workflow
simple_api/index
```
