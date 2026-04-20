# Data (Stage 1)

Stage 1 answers **"what data problem is this study solving?"** — once the design (Stage 0) has fixed the research identity and recipe shape, Stage 1 nails down the data source, the task, the evaluation window, the predictor universe, and the data-handling policies.

Stage 1 holds **20 meta axes** under the `1_data_task` layer, organised into five groups:

| # | Group | Axes | Focus |
|---|---|---:|---|
| §1.1 | [Source & Frame (1.1)](source.md) | 4 | Which dataset, at what frequency, within what information-set regime |
| §1.2 | [Task & Target (1.2)](task.md) | 4 | What is being forecast (task shape), how multi-step is produced, what forecast object, and how y_{t+h} is constructed |
| §1.3 | [Horizon & Evaluation Window (1.3)](horizon.md) | 4 | Training window size rule, training start, OOS regime filter, overlap handling |
| §1.4 | [Benchmark & Predictor Universe (1.4)](benchmark.md) | 4 | Benchmark families, predictor family, variable universe, deterministic components |
| §1.5 | Data handling policies (cleanup complete; impl pending) | 4 | Missing data policy, release lag, structural-break segmentation, contemporaneous X |

Stage 1 does not fix which model, which preprocessor, or which evaluation metric — those belong to Stage 2+ layers.

## Relation to Stage 0

The 6 meta axes of Stage 0 ([design](../design.md)) — `research_design`, `experiment_unit`, `axis_type`, `failure_policy`, `reproducibility_mode`, `compute_mode` — set the **grammar** of the study (which runner, what recipe shape, how parallel, how reproducible). Stage 1 fills that grammar with **content**: which data, which target, which horizons.

## Honest operational status

Layer 1 covers 20 axes. After the v0.9.3 Tier 1-3 drop and the §1.1 / §1.2 / §1.3 / §1.4 / §1.5 per-group walks (§1.5 cleanup PR in flight), the current breakdown is:

- **§1.1 Source & Frame** — fully honest. 4 axes, all values either operational or dropped.
- **§1.2 Task & Target** — fully operational (task / forecast_type / forecast_object / horizon_target_construction); horizon_target_construction applies as a metric-scale transform at the central row site.
- **§1.3 Horizon & Evaluation Window** — fully operational (min_train_size / training_start_rule / oos_period / overlap_handling); see horizon.md for per-axis semantics.
- **§1.4 Benchmark & Predictor Universe** — fully operational (benchmark_family / predictor_family / variable_universe / deterministic_components). 19 formerly-demoted values wired via leaf_config input channels + deterministic feature augmentation; 4 dropped values stay out of v1.0 scope.
- **§1.5 Data Handling Policies** — cleanup complete (6 axes dropped, 12 values dropped, 9 demoted to registry_only). 4 axes retained (missing_availability / release_lag_rule / structural_break_segmentation / contemporaneous_x_rule). Impl PR pending to flip the 9 demoted values operational via simple leaf_config input channels + break-dummy additions. evaluation_scale re-homed to Layer 2 (where it always belonged as a PreprocessContract field).

Each §1.x group document flags the honest status of every value it covers. Treat the current `design.md` / `data/*.md` state as the source of truth rather than the raw registry dump.

## Data source

Each of the three built-in datasets (FRED-MD, FRED-QD, FRED-SD) has its own documentation covering citation, download path, variable categories, transformation codes, and changes from the original working paper to the current vintage:

- [FRED-MD](datasets/fred_md.md) — monthly U.S. macro panel (McCracken & Ng 2016).
- [FRED-QD](datasets/fred_qd.md) — quarterly U.S. macro panel (McCracken & Ng 2020).
- [FRED-SD](datasets/fred_sd.md) — state-level real-time panel, mixed-frequency (Bokun, Jackson, Kliesen, Owyang 2022).

```{toctree}
:maxdepth: 1
:hidden:

source
task
horizon
benchmark
datasets/fred_md
datasets/fred_qd
datasets/fred_sd
```
