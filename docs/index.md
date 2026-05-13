# macroforecast

> What macroforecast does: run forecasting research on
> **FRED-MD / FRED-QD / FRED-SD** with your own dataset, preprocessing,
> and models — and benchmark them head-to-head against established methods.
> One YAML recipe defines the full study; `macroforecast.replicate(...)`
> regenerates every artifact identically from the recipe.

## Pick your path

::::{grid} 1 2 2 3
:gutter: 3
:class-container: pick-your-path

:::{grid-item-card} 🚀 Researchers
:link: for_researchers/index
:link-type: doc

Run a forecast on FRED data and read results.
:::

:::{grid-item-card} 📝 Recipe authors
:link: for_recipe_authors/index
:link-type: doc

Write a custom recipe with your model and preprocessor.
:::

:::{grid-item-card} 🛠 Contributors
:link: for_contributors/index
:link-type: doc

Modify the package source.
:::

:::{grid-item-card} 📐 Architecture
:link: architecture/index
:link-type: doc

Why layers and contracts are split this way — design narrative.
:::

:::{grid-item-card} 📖 Encyclopedia
:link: encyclopedia/index
:link-type: doc

Look up every recipe option (auto-generated, 189 pages).
:::

:::{grid-item-card} 🔁 Replications
:link: replications/index
:link-type: doc

Bundled studies and research walkthroughs.
:::

:::{grid-item-card} 🧭 Navigator
:link: navigator/index
:link-type: doc

Visually explore the layer and pipeline topology.
:::

:::{grid-item-card} 🆘 Troubleshooting
:link: troubleshooting
:link-type: doc

Errors, FAQs, common fixes.
:::

::::

> **Architecture vs Encyclopedia**: same 12-layer system, two angles.
> Architecture is **prose** — "why is L2 separated from L3", "how does
> L7 read L4 sinks", "what are the cross-layer references". Encyclopedia
> is **lookup** — one page per axis with every option's definition,
> when to use, when NOT, references, related options. Architecture is
> hand-written; encyclopedia is auto-generated from
> `LayerImplementationSpec` + `OPTION_DOCS` and locked by the ci-docs
> drift gate.

## Architecture overview

12-layer canonical design — see [architecture](architecture/index.md). The full
4-part design lives under `plans/design/` in the repo.

```text
L0 -> L1 -> L2 -> L3(pipeline) -> L4(pipeline) -> L5 -> L6 -> L7(pipeline) -> L8
        |      |      |       |
       L1.5   L2.5   L3.5    L4.5 diagnostics
```

## Install

```bash
pip install macroforecast
```

See [install](install.md) for extras and source install.

## License

MIT

```{toctree}
:hidden:
:maxdepth: 1

install
for_researchers/index
for_recipe_authors/index
for_contributors/index
architecture/index
navigator/index
replications/index
encyclopedia/index
troubleshooting
CONVENTIONS
```
