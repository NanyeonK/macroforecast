# macroforecast

> What you can do: run reproducible macro-forecasting experiments with custom
> preprocessing DAGs, 35+ models, statistical tests, importance interpretation,
> and FRED-SD geographic visualization. One YAML recipe → bit-exact replicable
> manifest, replayable bit-for-bit by `macroforecast.replicate(...)`.

## Pick your path

| If you want to ... | Start here |
|---|---|
| Run a forecast on FRED data and read the results | [Researchers](for_researchers/index.md) |
| Write a custom recipe with your model / preprocessor | [Recipe authors](for_recipe_authors/index.md) |
| Modify the package source / contribute | [Contributors](for_contributors/index.md) |
| Look up a specific axis / option / Python symbol | [Reference](reference/index.md) |
| See replication studies (bundled examples + 4 research walkthroughs) | [Replications](replications/index.md) |
| Visually explore the layer / DAG topology | [Navigator](navigator/index.md) |

## Architecture

12-layer canonical design — see [architecture](architecture/index.md). The full
4-part design lives under `plans/design/` in the repo.

```text
L0 -> L1 -> L2 -> L3(DAG) -> L4(DAG) -> L5 -> L6 -> L7(DAG) -> L8
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
reference/index
CONVENTIONS
```
