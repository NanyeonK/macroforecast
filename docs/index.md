# macroforecast

> What macroforecast does: run forecasting research on
> **FRED-MD / FRED-QD / FRED-SD** with your own dataset, preprocessing,
> and models — and benchmark them head-to-head against established methods.
> One YAML recipe defines the full study; `macroforecast.replicate(...)`
> regenerates every artifact identically from the recipe.

## Pick your path

| If you want to ... | Start here |
|---|---|
| Run a forecast on FRED data and read the results | [Researchers](for_researchers/index.md) |
| Write a custom recipe with your model / preprocessor (incl. partial-layer debugging and custom hooks) | [Recipe authors](for_recipe_authors/index.md) |
| Modify the package source / contribute | [Contributors](for_contributors/index.md) |
| **Understand the design** — why layers / sub-layers / contracts are split this way (read top-to-bottom) | [Architecture](architecture/index.md) — design narrative, ~18 prose pages |
| **Look up an option** — what does `apply_official_tcode` do, what models are L7 `shap_tree` compatible with, etc. (search / browse) | [Encyclopedia](encyclopedia/index.md) — auto-generated reference, 189 pages |
| See replication studies (bundled examples + 3 research walkthroughs today, 4+ more in v0.9.1) | [Replications](replications/index.md) |
| Visually explore the layer / DAG topology | [Navigator](navigator/index.md) |
| Hit an error or something doesn't work | [Troubleshooting & FAQ](troubleshooting.md) |

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
