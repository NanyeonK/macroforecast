# Execution Engine

The execution engine turns a compiled recipe into forecasts, metrics, and artifacts.

Current responsibilities include:

- raw data loading and cache selection
- optional FRED-SD component loading with FRED-MD/FRED-QD frequency resolution
- data-task transformations
- preprocessing
- model execution
- benchmark execution
- metric computation
- statistical tests
- importance artifacts
- manifest writing

This page should eventually document the final module boundaries after the engine is split into smaller components.

Target boundaries:

- orchestration
- data task
- preprocessing runner
- model runner
- benchmark runner
- metrics runner
- optional artifact runners
- artifact and manifest writer
