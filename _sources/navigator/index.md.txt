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
- how to generate a YAML recipe preview and run the compiled recipe.

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
2. Use the Navigator App to change path choices in the browser, inspect disabled branches, and preview the YAML diff.
3. Load a replication entry or import an existing recipe YAML when you want the tree state to start from a known route.
4. Download the generated YAML and run `macrocast-navigate resolve` to confirm authoritative compiler status before execution.
5. Run the resolved YAML with `macrocast-navigate run`.
6. Move to Detailed Docs or API Reference only when extending internals or writing custom plugins.

The browser app exports the same registry-backed rule metadata as the CLI through `navigator_state_engine_v1`. It is designed for fast path exploration: selections immediately update disabled branches, compatibility messages, resolver-preview status, and YAML preview output. The CLI resolver remains the authoritative compile check because it can see the local runtime, optional backends, and execution environment.

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
