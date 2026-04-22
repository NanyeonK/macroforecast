# Data (Stage 1)

Stage 1 answers **"what official data frame does this study start from?"** — once the design (Stage 0) has fixed the research identity and recipe shape, Stage 1 loads the dataset, fixes the information set, identifies the target/horizon/sample period, and applies only official availability rules.

After the layer-boundary migration, Stage 1 holds **9 canonical axes** under the `1_data_task` layer:

| Group | Axes | Focus |
|---|---|---:|---|
| Source & frame | 4 | `dataset`, `dataset_source`, `frequency`, `information_set_type` |
| Task identity | 1 | `task`; target/targets/horizons live in `leaf_config` |
| Availability and universe | 4 | `missing_availability`, `release_lag_rule`, `contemporaneous_x_rule`, `variable_universe` |

Stage 1 does not fix which model, benchmark, researcher preprocessing, or evaluation metric to use — those belong to Stage 2+ layers.

## Relation to Stage 0

The 6 meta axes of Stage 0 ([design](../design.md)) — `research_design`, `experiment_unit`, `axis_type`, `failure_policy`, `reproducibility_mode`, `compute_mode` — set the **grammar** of the study (which runner, what recipe shape, how parallel, how reproducible). Stage 1 fills that grammar with **content**: which data, which target, which horizons.

## Honest operational status

Layer 1 now covers 9 canonical axes. The migration moves model, benchmark, preprocessing, and inference choices out of Layer 1:

- **Kept in Layer 1** — dataset/source/frequency/information set, task identity, official availability, release lag, contemporaneous information rule, raw variable universe.
- **Moved to Layer 2** — target representation and deterministic/break features.
- **Moved to Layer 3** — benchmark, forecast type/object, predictor family, training-window rules.
- **Moved to Layer 4** — OOS regime subset evaluation.
- **Moved to Layer 6** — overlap/HAC inference handling.

During migration, some detailed pages still document the old historical grouping. Treat `docs/detail/layer_boundary_contract.md` and `docs/detail/layer_axis_migration_plan.md` as the source of truth.

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
policies
datasets/fred_md
datasets/fred_qd
datasets/fred_sd
```
