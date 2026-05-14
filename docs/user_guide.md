# User guide

Run forecasts on **FRED-MD / FRED-QD / FRED-SD** or your own data. The only
structural difference between consuming the package and customizing it is
whether you register a Python callable.

## Two paths

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 1. Using the package
:link: for_researchers/simple_api/index
:link-type: doc

After the [Getting started](getting_started.md) intro: the high-level
Simple API, the full Recipe (YAML) grammar, and FRED dataset references.
:::

:::{grid-item-card} 2. Bring your own
:link: for_researchers/user_data_workflow
:link-type: doc

Plug in your own dataset (CSV / Parquet), model, preprocessor, or target
transformer.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2
:caption: 1. Using the package

for_researchers/simple_api/index
for_researchers/simple_api/quickstart
for_researchers/simple_api/run_experiment
for_researchers/simple_api/compare_models
for_researchers/simple_api/read_results
for_researchers/simple_api/fred_sd
for_researchers/recipe_gallery
for_recipe_authors/design
for_recipe_authors/data_policies
for_recipe_authors/data/index
for_recipe_authors/default_profiles
for_researchers/runtime_support
for_researchers/understanding_output
for_researchers/fred_datasets/index
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: 2. Bring your own

for_researchers/user_data_workflow
for_recipe_authors/custom_function_quickstart
for_recipe_authors/custom_hooks
for_recipe_authors/target_transformer
for_recipe_authors/partial_layer_execution
```
