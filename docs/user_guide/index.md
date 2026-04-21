# User Guide

In-depth documentation for every macrocast axis whose per-axis walk is complete.

## Stage 0 — Design

[Design (Stage 0)](design.md) — six axes that decide the shape of the study:

- `research_design` — single forecast vs. horse race vs. bundle vs. replication.
- `experiment_unit` — which runner owns the recipe (auto-derived).
- `axis_type` — fixed / sweep / nested / conditional / derived per axis.
- `failure_policy` — how to handle per-cell failures.
- `reproducibility_mode` — how strict to make determinism.
- `compute_mode` — which level of the sweep runs in parallel.

## Stage 1 — Data

[Data (Stage 1)](data/index.md) — twenty axes, organised into five groups:

- [1.1 Source & Frame](data/source.md) — `dataset`, `dataset_source`, `frequency`, `information_set_type`. Each FRED dataset is documented in depth under [Sources](../sources/index.md).
- [1.2 Task & Target](data/task.md) — `task`, `forecast_type`, `forecast_object`, `horizon_target_construction`.
- [1.3 Horizon & Evaluation Window](data/horizon.md) — `min_train_size`, `training_start_rule`, `oos_period`, `overlap_handling`.
- [1.4 Benchmark & Predictor Universe](data/benchmark.md) — `benchmark_family`, `predictor_family`, `variable_universe`, `deterministic_components`.
- [1.5 Data Handling Policies](data/policies.md) — `missing_availability`, `release_lag_rule`, `structural_break_segmentation`, `contemporaneous_x_rule`.

## Stages 2 through 7

Not yet in the user-facing docs. Plan files live under `plans/` in the repo; docs land when each per-axis walk completes.

```{toctree}
:hidden:
:maxdepth: 1
:caption: Stages

design
data/index
```
