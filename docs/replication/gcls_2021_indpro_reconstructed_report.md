# GCLS 2021 INDPRO Reconstructed Replication Report

This report records the corrected `macroforecast` run for the INDPRO cells in
Table 2 of Goulet Coulombe, Leroux, Stevanovic, and Surprenant (2021),
"Macroeconomic data transformations matter," *International Journal of
Forecasting*, 37(4), 1338-1354.

Report date:
: 2026-06-06

Replication level:
: `reconstructed_design`, not `exact_table_replication`.

Execution scope:
: INDPRO only, six horizons, paper Table 2 best-specification cells, compared
  against the matching factor-model benchmark on the same realized-target
  support.

Main result:
: the corrected INDPRO best-specification cells beat the matching FM benchmark
  at every horizon. Mean relative RMSE is `0.889818`; median relative RMSE is
  `0.935358`. Relative RMSE below `1` means the Table 2 best-specification cell
  has lower RMSE than the FM benchmark on the common support.

## Status

| Item | Status |
| --- | --- |
| Best-spec INDPRO run | complete |
| FM benchmark run | complete |
| Common-support comparison | complete |
| Failed tasks | 0 |
| Active server jobs | none detected |
| Table-identical replication claim | not made |

The run should be read as corrected package evidence. It is not a claim that the
numbers are identical to the paper's Table 2, because the checked paper,
appendix, local files, and public author materials do not expose a full
machine-readable replication package, exact FRED-MD vintage, or exact MATLAB
backend state.

## Source Material

Sources used to define the replication setting:

- IJF article DOI: <https://doi.org/10.1016/j.ijforecast.2021.05.005>
- arXiv working-paper page: <https://arxiv.org/abs/2008.01714>
- local main PDF:
  `/Users/nanyeon/Library/CloudStorage/SynologyDrive-second_brain/wiki/raw/papers/10.1016j.ijforecast.2021.05.005.pdf`
- local appendix PDF:
  `/Users/nanyeon/Library/CloudStorage/SynologyDrive-second_brain/wiki/raw/papers/10.1016j.ijforecast.2021.05.005_appendix.pdf`
- local review note:
  `/Users/nanyeon/Library/CloudStorage/SynologyDrive-second_brain/wiki/papers/reviews/10.1016j.ijforecast.2021.05.005-ea1152c5.md`
- author MARX snippet:
  `/Users/nanyeon/Library/CloudStorage/SynologyDrive-second_brain/wiki/raw/paper_code/coulombe_site_github_20260530/marx/MARX_cheap_code.R`

## Execution Artifacts

The run was executed on server1.

| Object | Path |
| --- | --- |
| Source checkout used for run | `/home/nanyeon99/project/macroforecast_gcls_replication_main_2f526bdf` |
| Source checkout commit | `2f526bdf` |
| Best-spec output root | `/home/nanyeon99/project/macroforecast_gcls_runs/table2_indpro_full_20260605` |
| FM benchmark output root | `/home/nanyeon99/project/macroforecast_gcls_runs/indpro_fm_benchmark_20260606` |
| Relative comparison output root | `/home/nanyeon99/project/macroforecast_gcls_runs/indpro_relative_vs_fm_20260606` |
| Relative comparison CSV | `/home/nanyeon99/project/macroforecast_gcls_runs/indpro_relative_vs_fm_20260606/indpro_relative_vs_fm.csv` |

The best-spec run produced about `19M` of output, the FM benchmark about `16M`,
and the relative comparison about `336K`.

## Data And Sample

| Axis | Setting |
| --- | --- |
| Dataset | FRED-MD |
| Vintage | `2018-01` |
| Loader | `mf.data.load_fred_md(vintage="2018-01")` |
| Raw panel | 708 monthly rows x 127 columns |
| Raw period | `1959-01` through `2017-12` |
| Preprocessing | official McCracken-Ng FRED-MD t-code pipeline |
| Processed panel | 706 monthly rows x 127 columns |
| Processed period | `1959-03` through `2017-12` |
| Initial estimation start | `1960-01` |
| Test calendar | monthly origins from `1980-01` through `2017-12` where realized targets are available |
| Horizons | `1, 3, 6, 9, 12, 24` months |
| Target | `INDPRO` |

The h-step forecast at origin `t` is scored only when the realized target dated
`t + h` is available. For h=24 with a `2018-01` FRED-MD vintage ending in
`2017-12`, the final scored origins stop before the tail origins whose
realizations would fall after the vintage endpoint. This is expected and is not
a missing monthly-step bug: monthly origins still move by one month, but scoring
requires the future realized target.

## Best-Specification Cells

The batch script fixes the Table 2 best-specification cell for each INDPRO
horizon:

| Horizon | Target policy | Model | Feature case |
| ---: | --- | --- | --- |
| 1 | `direct_average` | `random_forest` | `F-X-MARX-Level` |
| 3 | `direct_average` | `random_forest` | `MARX` |
| 6 | `path_average` | `random_forest` | `MARX` |
| 9 | `path_average` | `random_forest` | `MARX` |
| 12 | `path_average` | `random_forest` | `MARX` |
| 24 | `direct_average` | `random_forest` | `F-Level` |

Random forest was run with `n_estimators=200`, `min_samples_leaf=5`,
`max_features=1/3`, `bootstrap=True`, `random_state=123`, and `n_jobs=1`.
Hyperparameter tuning was off for this pass, so this is a fixed paper-style
configuration rather than a full appendix optimizer replication.

## Command Log

Best-spec INDPRO batch:

```bash
uv run python scripts/replication/gcls_2021_table2_batch.py \
  --out-root /home/nanyeon99/project/macroforecast_gcls_runs/table2_indpro_full_20260605 \
  --targets INDPRO \
  --workers 3 \
  --vintage 2018-01 \
  --cache-root /home/nanyeon99/project/macroforecast_replication_cache \
  --start-year 1980 \
  --end-year 2017 \
  --n-estimators 200 \
  --random-state 123 \
  --tuning-mode off \
  --skip-existing
```

Observed batch summary:

```text
status: done
workers: 3
task_count: 6
finished_count: 6
failed_count: 0
elapsed: about 15.4 hours
```

The matching FM benchmark used the same single-cell runner with
`--feature-case F --model far`, horizon-specific target policies matching the
best-spec cell, and the same `2018-01` vintage, `1980` to `2017` calendar, and
target construction.

Conceptually:

```bash
uv run python scripts/replication/gcls_2021_table2_single.py \
  --target-alias INDPRO \
  --horizon <horizon> \
  --feature-case F \
  --target-policy <matching_policy> \
  --model far \
  --vintage 2018-01 \
  --cache-root /home/nanyeon99/project/macroforecast_replication_cache \
  --out-dir /home/nanyeon99/project/macroforecast_gcls_runs/indpro_fm_benchmark_20260606/<task_slug> \
  --start-year 1980 \
  --end-year 2017 \
  --random-state 123 \
  --tuning-mode off \
  --skip-existing
```

The relative comparison aligns best-spec and FM forecast files by realized
target date, checks that the realized targets are identical, and computes RMSE
and relative MSE/RMSE on the common support.

## Absolute Results

| Horizon | Best-spec task | Rows | RMSE | MAE |
| ---: | --- | ---: | ---: | ---: |
| 1 | `INDPRO_h1_direct_average_random_forest_F-X-MARX-Level` | 455 | 0.005964 | 0.004248 |
| 3 | `INDPRO_h3_direct_average_random_forest_MARX` | 453 | 0.004482 | 0.003086 |
| 6 | `INDPRO_h6_path_average_random_forest_MARX` | 450 | 0.003937 | 0.002727 |
| 9 | `INDPRO_h9_path_average_random_forest_MARX` | 447 | 0.003559 | 0.002487 |
| 12 | `INDPRO_h12_path_average_random_forest_MARX` | 444 | 0.003328 | 0.002316 |
| 24 | `INDPRO_h24_direct_average_random_forest_F-Level` | 432 | 0.002407 | 0.001698 |

The row counts fall with the horizon because later origins need later realized
targets. The h=24 row count is `432`, corresponding to the available common
support after excluding tail origins whose 24-month-ahead target is unavailable
in the vintage.

## Relative Results Against FM

| Horizon | Best RMSE | FM RMSE | Relative MSE | Relative RMSE | Common rows | Beats FM |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.005964 | 0.006283 | 0.900921 | 0.949169 | 455 | yes |
| 3 | 0.004482 | 0.004573 | 0.960364 | 0.979982 | 453 | yes |
| 6 | 0.003937 | 0.004159 | 0.896344 | 0.946754 | 450 | yes |
| 9 | 0.003559 | 0.003852 | 0.853704 | 0.923961 | 447 | yes |
| 12 | 0.003328 | 0.003752 | 0.786731 | 0.886979 | 444 | yes |
| 24 | 0.002407 | 0.003692 | 0.425190 | 0.652066 | 432 | yes |

Common-support checks:

```text
actual_max_abs_diff: 0.0 for every horizon
invalid_rows: 0
nan_prediction_rows: 0
nan_actual_rows: 0
```

Interpretation:

- `actual_max_abs_diff=0.0` means the best-spec and FM rows use the same
  realized target values after alignment.
- `relative_mse < 1` means the best-spec cell has lower squared-error loss than
  FM.
- `relative_rmse < 1` means the same result expressed in RMSE units.
- The largest improvement appears at h=24, where the RF `F-Level`
  direct-average cell has relative RMSE `0.652066`.

## What Changed Relative To The Invalid Diagnostic Runs

Earlier package diagnostics were useful for finding defects, but they are not
valid replication evidence. The corrected run differs in the following
material ways:

| Issue found in earlier diagnostics | Corrected behavior |
| --- | --- |
| h-step labels were allowed when the realized target date was after the forecast origin support | forecasts are scored only when `t + h` is available |
| `average_change` was applied to an already McCracken-Ng transformed target | direct-average cells use `average_value`; path-average cells use one-step `value` targets |
| target-derived paper blocks were missing | `MARX_y` and `MAF_y` can be built from `input="target_panel"` |
| feature materialization was too slow for repeated windows | runner now supports cached/corrected feature construction paths |
| invalid runs compared diagnostic shortcuts | corrected run compares best-spec cells against matching FM cells on identical actual support |

## Remaining Replication Gaps

These gaps are not package runtime failures; they are evidence boundaries for
claiming exact paper-table equality.

| Gap | Consequence |
| --- | --- |
| Exact FRED-MD vintage is not stated in the checked materials | `2018-01` is the first defensible post-`2017M12` candidate, but may not be the paper's exact vintage |
| Full machine-readable replication package was not found | exact table reproduction cannot be audited line-by-line against author code |
| MATLAB tree and optimizer defaults are not exactly portable | Python/scikit-style RF/BT values can differ even under the same high-level algorithm |
| This pass uses `tuning-mode=off` | appendix GA/Bayesian/random-CV tuning is not yet replicated for every learner |
| Benchmark is fixed FM mapping | BIC-selected FM variants should be added if the paper's benchmark implementation is recovered |
| This report covers INDPRO only | ten-target Table 2 completion still requires EMP, UNRATE, INCOME, CONS, RETAIL, HOUST, M2, CPI, and PPI |

## Next Actions

1. Run the same corrected pipeline for the remaining nine Table 2 targets.
2. Add a benchmark helper that can switch between fixed FM and BIC-selected FM
   once the paper's exact benchmark selection rule is pinned down.
3. Run at least one `paper-small` tuning pass for Elastic Net, Adaptive Lasso,
   Linear Boosting, and Boosted Trees to verify the tuned-learner branch.
4. Add paper-table capture comparison in the notebook page after the full
   ten-target table is available.
5. Keep the invalid diagnostic section in the setting page as a debugging log,
   but do not use those values as evidence.

