# macrocast

> Given a standardized macro dataset adapter and a fixed forecasting recipe, compare forecasting tools under identical information set, sample split, benchmark, and evaluation protocol.

## Current capabilities

- **Layer-contract runtime**: L1-L8 plus L1.5-L4.5 diagnostic artifacts
  materialize through `macrocast.core.runtime.execute_minimal_forecast`.
- **Data**: custom panels and official FRED-MD/FRED-QD raw adapters are
  supported in the core path; FRED-SD is available in the broader execution
  engine and remains provisional in the core path.
- **Models**: the core runtime executes expanding-window sklearn linear
  models (`ols`, `ridge`, `lasso`, `elastic_net`) and propagates benchmark
  flags. The broader registry contains additional model families for schema
  validation and legacy execution paths.
- **Evaluation**: point metrics, benchmark-relative metrics, rankings, L6
  lightweight statistical tests, L7 basic importance, and L8 file export are
  materialized.
- **Diagnostics**: L1.5-L4.5 generate JSON diagnostic artifacts. Figure/table
  rendering is still a specialized runtime task.

See `docs/getting_started/runtime_support.md` for the current support matrix.

## Structure

```
macrocast/       # package source
  raw/           # dataset loading
  registry/      # legacy per-axis choice registry
  core/          # layer-contract schema and runtime surface
  compiler/      # recipe -> execution bridge
  execution/     # runtime engine
tests/           # regression and layer-contract tests
docs/            # public documentation
plans/           # internal planning
```

## Development rules

- `docs/` is public-facing only; internal planning stays in `plans/`
- Every new code surface must be documented in `docs/`
- Grammar is fixed before registry content is expanded
- One path = one fully specified forecasting study
- `macrocast/core` is the source of truth for the layer-contract DAG system;
  `macrocast/registry` remains for backward compatibility.
