# Execution Engine

macroforecast currently has two execution surfaces.

## Core Layer-Contract Runtime

`macroforecast.core.runtime.execute_minimal_forecast` executes the L1-L8 layer-contract path used by the new schema.

Current responsibilities:

- L1 data materialization for custom panels and official FRED-MD/FRED-QD raw adapters
- L2 preprocessing materialization
- L3 deterministic feature DAG execution
- L4 expanding-window sklearn linear model execution
- L5 point metric and benchmark-relative metric computation
- L1.5-L4.5 diagnostic artifact materialization
- L6 lightweight statistical-test artifact materialization
- L7 basic importance artifact materialization
- L8 output directory and manifest writing

The core runtime is the preferred path for validating new layer specs and for reproducible linear-model smoke studies.

## Legacy Experiment Engine

`macroforecast.compiler` and `macroforecast.execution` continue to support older recipe surfaces and broader experiment tests. Some older docs and examples still refer to this path. Those outputs may use legacy filenames such as `predictions.csv`, `metrics.json`, or `artifact_manifest.json`.

## Current Boundaries

The core runtime intentionally does not yet implement every schema-visible method. Advanced tree/deep/VAR model execution, exact bootstrap statistical tests, SHAP backends, rendered figures, Parquet/LaTeX/HTML export, compression, and replication APIs remain specialized runtime work.

See [Runtime Support Matrix](../getting_started/runtime_support.md) for the current per-layer boundary.

## Target Boundaries

Long term, runtime code should split into smaller modules:

- orchestration
- data loading
- preprocessing runner
- feature runner
- model runner
- metrics runner
- statistical-test runner
- importance runner
- diagnostic renderer
- artifact and manifest writer

The current `macroforecast.core.runtime` file is a bridge: it keeps the layer-contract path executable while those module boundaries mature.
