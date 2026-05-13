# User guide

Run forecasts on **FRED-MD / FRED-QD / FRED-SD** or your own data. The only
structural difference between consuming the package and customizing it is
whether you register a Python callable.

## Two paths

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 1. Using the package
:link: for_researchers/quickstart
:link-type: doc

Run forecasts with the bundled FRED panels and the 35+ built-in models. Read
the results, sweep recipes, look up support boundaries.
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

for_researchers/quickstart
for_researchers/first_study
for_researchers/simple_api/index
for_researchers/recipe_gallery
for_researchers/runtime_support
for_researchers/understanding_output
for_researchers/fred_datasets/index
for_recipe_authors/design
for_recipe_authors/data_policies
for_recipe_authors/data/index
for_recipe_authors/default_profiles
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
