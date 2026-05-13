# User guide

Run forecasts on **FRED-MD / FRED-QD / FRED-SD** or your own data.
The workflow is the same whether you only consume the bundled models or also
plug in your own — the only structural difference is whether you register a
Python callable.

## Chapters

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 1. Get started
:link: for_researchers/quickstart
:link-type: doc

Minimal recipe, simple API, your first complete study, runtime support, output guide.
:::

:::{grid-item-card} 2. Bring your own data
:link: for_researchers/user_data_workflow
:link-type: doc

Custom CSV / Parquet panels, merge with FRED-MD / QD / SD, common pitfalls.
:::

:::{grid-item-card} 3. Plug in custom code
:link: for_recipe_authors/custom_function_quickstart
:link-type: doc

Register your own model, preprocessor, or target transformer.
:::

:::{grid-item-card} 4. Recipe grammar
:link: for_recipe_authors/design
:link-type: doc

Layer-contract design, sweep axes, partial-layer execution, default profiles.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2
:caption: 1. Get started

for_researchers/quickstart
for_researchers/first_study
for_researchers/simple_api/index
for_researchers/runtime_support
for_researchers/understanding_output
for_researchers/recipe_gallery
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: 2. Bring your own data

for_researchers/user_data_workflow
for_researchers/fred_datasets/index
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: 3. Plug in custom code

for_recipe_authors/custom_function_quickstart
for_recipe_authors/custom_hooks
for_recipe_authors/target_transformer
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: 4. Recipe grammar

for_recipe_authors/design
for_recipe_authors/data_policies
for_recipe_authors/data/index
for_recipe_authors/partial_layer_execution
for_recipe_authors/default_profiles
```
