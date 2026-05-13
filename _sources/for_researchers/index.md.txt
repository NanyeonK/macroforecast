# For researchers

You want to run a macro-forecasting study, get the artifacts, and read the
results. You do **not** want to author new model classes or modify the package
source. Start here.

## If you want to ...

| Goal | Page |
|---|---|
| Set up the package and run your first recipe | [Quickstart](quickstart.md) |
| Build your first study with a working recipe template | [First study](first_study.md) |
| See what the runtime executes today vs. what is schema-only | [Runtime support matrix](runtime_support.md) |
| Understand the output directory and manifest layout | [Understanding output](understanding_output.md) |
| Look up FRED-MD / FRED-QD / FRED-SD column dictionaries | [FRED datasets](fred_datasets/index.md) |
| Check FRED-SD T-code policy (transform-code defaults for state series) | [FRED-SD transform policy](fred_datasets/fred_sd/transform_policy.md), [inferred T-codes](fred_datasets/fred_sd/inferred_tcodes.md), [v0.1 review](fred_datasets/fred_sd/inferred_tcode_review_v0_1.md) |
| Browse the 38 bundled recipes (with extras + smoke-test status) | [Recipe gallery](recipe_gallery.md) |
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
runtime_support
understanding_output
fred_datasets/index
recipe_gallery
user_data_workflow
simple_api/index
```
