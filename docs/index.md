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

:::{grid-item-card} 🚀 Getting started
:link: getting_started
:link-type: doc

Install, quickstart, your first complete study.
:::

:::{grid-item-card} 📝 User guide
:link: user_guide
:link-type: doc

Recipe authoring, custom models, FRED & custom datasets.
:::

:::{grid-item-card} 📖 Reference
:link: reference
:link-type: doc

Encyclopedia of every option + architecture design narrative.
:::

:::{grid-item-card} 🔁 Replications
:link: replications
:link-type: doc

Paper replications, recipe gallery, layer navigator.
:::

:::{grid-item-card} 🆘 Help
:link: help
:link-type: doc

Troubleshooting, contributing, conventions.
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

getting_started
user_guide
reference
replications
help
```
