# Navigator Docs

The navigator is the primary documentation surface for choosing a macrocast study path. It is built for packages where the number of valid combinations is large enough that API documentation alone becomes misleading.

Open the interactive app:

```{raw} html
<p><a class="reference external" href="../navigator_app/index.html">Open Navigator App</a></p>
```

The navigator answers these questions before execution:

- which options are selectable at the current node;
- which options are disabled;
- why each disabled option is disabled;
- how each selection changes the canonical path;
- which paper-style replication paths are close to the current route;
- how to generate a YAML recipe and run it.

## Pages

| Page | Purpose |
|---|---|
| [Tree Navigator](tree_navigator.md) | Show selectable and disabled branches for every Layer 0-7 axis. |
| [Path Resolver](path_resolver.md) | Compile a YAML path and show execution status, warnings, blocked reasons, and capability matrix. |
| [Compatibility Engine](compatibility_engine.md) | Explain constraint rules such as model/feature compatibility, forecast-object metrics, and importance-method restrictions. |
| [Replication Library](replication_library.md) | Start from known paper-style routes, inspect exact paths, generate YAML, and understand deviations. |
| [YAML and Execution](yaml_execution.md) | Save a selected path as YAML, run it from CLI, and reproduce the same route in a notebook. |

## Recommended Flow

1. Start from a default recipe, a replication entry, or a hand-written YAML.
2. Run `macrocast-navigate tree` to inspect current choices and disabled branches.
3. Run `macrocast-navigate resolve` to confirm compiler status.
4. Generate or edit the YAML.
5. Run the YAML with `macrocast-navigate run`.
6. Move to Detailed Docs or API Reference only when extending internals or writing custom plugins.

## Boundary

Layers 0-3 are forecast-construction contracts. Bad choices here can prevent forecast generation or create leakage, so the navigator is strict.

Layers 4-7 consume forecast artifacts: evaluation, output, statistical tests, and interpretation. These layers still have contracts, but they do not build the forecast representation.

```{toctree}
:maxdepth: 1

tree_navigator
path_resolver
compatibility_engine
replication_library
yaml_execution
```
