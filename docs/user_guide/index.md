# User Guide

In-depth documentation for every macrocast axis whose per-axis walk is complete.

## Stage 0 — Design

[Design (Stage 0)](design.md) — axes that decide the execution grammar:

- `study_scope` — one target or multiple targets, with one fixed method path or a controlled method comparison.
- `axis_type` — fixed / sweep / nested / conditional / derived per axis.
- `failure_policy` — how to handle per-cell failures.
- `reproducibility_mode` — how strict to make determinism.
- `compute_mode` — serial by default, or a Study Scope-compatible parallel work layout.

## Stage 1 — Data

[Data (Stage 1)](data/index.md) — thirteen canonical axes after the layer-boundary migration, with historical/migrated detail pages kept for reference:

- [1.1 Source & Frame](data/source.md) — `dataset`, `source_adapter`, `frequency`, `information_set_type`, `official_transform_policy`, `official_transform_scope`.
- [1.2 Target Structure](data/target_structure.md) — `target_structure`.
- [1.3 Horizon & Evaluation Window](data/horizon.md) — `min_train_size`, `training_start_rule`, `oos_period`, `overlap_handling`.
- [1.4 Benchmark & Predictor Universe](data/benchmark.md) — `benchmark_family`, `predictor_family`, `variable_universe`, `deterministic_components`.
- [1.5 Data Handling Policies](data/policies.md) — `missing_availability`, `raw_missing_policy`, `raw_outlier_policy`, `release_lag_rule`, `contemporaneous_x_rule`.

## Stages 2 through 7

Not yet in the user-facing docs. Plan files live under `plans/` in the repo; docs land when each per-axis walk completes.

```{toctree}
:hidden:
:maxdepth: 1
:caption: Stages

design
data/index
```
