# Data (Stage 1)

Stage 1 answers **"what data problem is this study solving?"** — once the design (Stage 0) has fixed the research identity and recipe shape, Stage 1 nails down the data source, the task, the evaluation window, the predictor universe, and the data-handling policies.

Stage 1 holds **26 meta axes** under the `1_data_task` layer, organised into five groups:

| # | Group | Axes | Focus |
|---|---|---:|---|
| §1.1 | [Source & Frame (1.1)](source.md) | 4 | Which dataset, at what frequency, within what information-set regime |
| §1.2 | [Task & Target (1.2)](task.md) | 4 | What is being forecast (task shape), how multi-step is produced, what forecast object, and how y_{t+h} is constructed |
| §1.3 | [Horizon & Evaluation Window (1.3)](horizon.md) | 4 | Training window size rule, training start, OOS regime filter, overlap handling |
| §1.4 | Benchmark & predictor universe (coming) | 4 | Benchmark families, predictor family, variable universe |
| §1.5 | Data handling policies (coming) | 10 | Alignment, missing, release lag, vintage, break segmentation, etc. |

Stage 1 does not fix which model, which preprocessor, or which evaluation metric — those belong to Stage 2+ layers.

## Relation to Stage 0

The 6 meta axes of Stage 0 ([design](../design.md)) — `research_design`, `experiment_unit`, `axis_type`, `failure_policy`, `reproducibility_mode`, `compute_mode` — set the **grammar** of the study (which runner, what recipe shape, how parallel, how reproducible). Stage 1 fills that grammar with **content**: which data, which target, which horizons.

## Honest operational status

Layer 1 covers 26 axes. After the v0.9.3 Tier 1-3 drop and the §1.1 / §1.2 / §1.3 per-group walks, the current breakdown is:

- **§1.1 Source & Frame** — fully honest. 4 axes, all values either operational or dropped.
- **§1.2 Task & Target** — fully operational (task / forecast_type / forecast_object / horizon_target_construction); horizon_target_construction applies as a metric-scale transform at the central row site.
- **§1.3 Horizon & Evaluation Window** — fully operational (min_train_size / training_start_rule / oos_period / overlap_handling); see horizon.md for per-axis semantics.
- **§1.4 / §1.5** (pending) — still contain the older mix of truly-wired, metadata-only flow-through, and hollow-dispatch values. The per-group walks for these sections will land in follow-up PRs.

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
datasets/fred_md
datasets/fred_qd
datasets/fred_sd
```
