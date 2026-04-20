# Data (Stage 1)

Stage 1 answers **"what data problem is this study solving?"** — once the design (Stage 0) has fixed the research identity and recipe shape, Stage 1 nails down the data source, the task, the evaluation window, the predictor universe, and the data-handling policies.

Stage 1 holds **29 meta axes** under the `1_data_task` layer, organised into five groups:

| # | Group | Axes | Focus |
|---|---|---:|---|
| §1.1 | [Source & Frame (1.1)](source.md) | 4 | Which dataset, at what frequency, within what information-set regime |
| §1.2 | [Task & Target (1.2)](task.md) | 4 | What is being forecast (task shape), how multi-step is produced, what forecast object, and how y_{t+h} is constructed |
| §1.3 | Horizon & evaluation window (coming) | 7 | Horizons, OOS period shape, training-window rules |
| §1.4 | Benchmark & predictor universe (coming) | 4 | Benchmark families, predictor family, variable universe |
| §1.5 | Data handling policies (coming) | 10 | Alignment, missing, release lag, vintage, break segmentation, etc. |

Stage 1 does not fix which model, which preprocessor, or which evaluation metric — those belong to Stage 2+ layers.

## Relation to Stage 0

The 6 meta axes of Stage 0 ([design](../design.md)) — `research_design`, `experiment_unit`, `axis_type`, `failure_policy`, `reproducibility_mode`, `compute_mode` — set the **grammar** of the study (which runner, what recipe shape, how parallel, how reproducible). Stage 1 fills that grammar with **content**: which data, which target, which horizons.

## Honest operational status

Layer 1 covers 29 axes. After the v0.9.3 Tier 1-3 drop and the ongoing §1.1+ cleanup, the current breakdown is:

- **Truly wired** (value triggers a distinct runtime branch) — the majority of operational values in `dataset`, `information_set_type`, `benchmark_family`, `release_lag_rule`, `variable_universe` (for the 2 real values), `min_train_size`, and `structural_break_segmentation`.
- **Metadata-only flow-through** — the compiler reads the value into the manifest but no downstream code branches on it. Applies to most of `alignment_rule`, `horizon_target_construction`, `own_target_lags`, `horizon_list` named variants, several `warmup_rule` / `training_start_rule` values.
- **Hollow dispatch** — a switch block exists but every branch returns the input unchanged. Notably: `missing_availability` (7 of 8 values pass-through), `variable_universe` (6 of 9 values pass-through).
- **registry_only / future** — reserved for v1.1 / v2 adapters that require their own infrastructure phases.

Each §1.x group document flags the honest status of every value it covers. Values labelled `operational` in v1.0 without real runtime effect are being progressively demoted as the per-group walk proceeds; treat the current `design.md` / `data/*.md` state as the source of truth rather than the raw registry dump.

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
datasets/fred_md
datasets/fred_qd
datasets/fred_sd
```
